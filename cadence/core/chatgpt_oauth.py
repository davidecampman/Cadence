"""ChatGPT OAuth 2.1 + PKCE integration.

Allows users to authenticate with their ChatGPT subscription (Plus/Pro)
and use OpenAI Codex models at their subscription's flat rate instead of
per-token API billing.

The OAuth flow follows the PKCE (Proof Key for Code Exchange) pattern:
1. Generate a verifier/challenge pair and redirect to OpenAI's auth server
2. User authorizes in browser, gets redirected back with an auth code
3. Exchange the code for access + refresh tokens
4. Tokens are encrypted at rest using the existing Fernet keystore
5. Tokens auto-refresh when expired

Modeled after the approach used by OpenClaw and Cline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from typing import Any

import httpx

from cadence.core.keystore import _fernet, _ensure_data_dir, _DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI OAuth endpoints
# ---------------------------------------------------------------------------

OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_REVOKE_URL = "https://auth.openai.com/oauth/revoke"

# The public client ID from the official OpenAI Codex CLI.  This is the same
# ID used by Cline, Codebuff, and every other third-party PKCE integration —
# OpenAI does not offer a separate registration flow for third-party clients.
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Scopes matching the official Codex CLI flow.
# "offline_access" is required to receive a refresh_token.
DEFAULT_SCOPES = "openid profile email offline_access"

# Where we persist encrypted OAuth credentials
_OAUTH_PATH = _DATA_DIR / "chatgpt_oauth.enc"

# Default local callback matching the Codex CLI's registered redirect pattern.
# OpenAI whitelists "http://localhost:{port}/auth/callback" for this client ID.
DEFAULT_CALLBACK_PORT = 5173
DEFAULT_CALLBACK_URL = "http://localhost:5173/auth/callback"

# ---------------------------------------------------------------------------
# Codex API endpoint (different from the regular OpenAI API)
# ---------------------------------------------------------------------------
# When authenticated via ChatGPT OAuth, the Codex CLI uses a special endpoint
# at chatgpt.com, NOT api.openai.com.  The Responses API format is used
# (input + instructions), not Chat Completions (messages).
CODEX_API_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"

# Default Codex model available via subscription OAuth
CODEX_DEFAULT_MODEL = "gpt-5.3-codex"


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge (S256).

    Returns (verifier, challenge).
    """
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _generate_state() -> str:
    """Generate a random state parameter for CSRF protection."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Token persistence (encrypted at rest, same master key as keystore)
# ---------------------------------------------------------------------------

def _load_oauth_store() -> dict[str, Any]:
    """Load and decrypt the OAuth credential store."""
    if not _OAUTH_PATH.exists():
        return {}
    try:
        cipher = _fernet()
        encrypted = _OAUTH_PATH.read_bytes()
        decrypted = cipher.decrypt(encrypted)
        return json.loads(decrypted)
    except Exception:
        logger.warning("Failed to decrypt OAuth store; returning empty.")
        return {}


def _save_oauth_store(store: dict[str, Any]) -> None:
    """Encrypt and persist the OAuth credential store."""
    _ensure_data_dir()
    cipher = _fernet()
    plaintext = json.dumps(store).encode()
    _OAUTH_PATH.write_bytes(cipher.encrypt(plaintext))
    try:
        os.chmod(_OAUTH_PATH, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API — OAuth flow management
# ---------------------------------------------------------------------------

class ChatGPTOAuthState:
    """In-memory state for an in-progress OAuth flow."""

    def __init__(self):
        self.verifier: str | None = None
        self.challenge: str | None = None
        self.state: str | None = None

    def clear(self):
        self.verifier = None
        self.challenge = None
        self.state = None


# Singleton for the current in-flight OAuth flow
_pending_flow = ChatGPTOAuthState()


def build_authorize_url(
    callback_url: str = DEFAULT_CALLBACK_URL,
    scopes: str = DEFAULT_SCOPES,
) -> str:
    """Start a new OAuth flow: generate PKCE pair, state, and return the
    authorization URL the user should open in their browser.
    """
    verifier, challenge = _generate_pkce_pair()
    state = _generate_state()

    _pending_flow.verifier = verifier
    _pending_flow.challenge = challenge
    _pending_flow.state = state

    params = {
        "response_type": "code",
        "client_id": OPENAI_CLIENT_ID,
        "redirect_uri": callback_url,
        "scope": scopes,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    qs = "&".join(f"{k}={_url_encode(v)}" for k, v in params.items())
    return f"{OPENAI_AUTH_URL}?{qs}"


def _url_encode(value: str) -> str:
    """Minimal percent-encoding for query string values."""
    from urllib.parse import quote
    return quote(value, safe="")


async def exchange_code(
    code: str,
    state: str,
    callback_url: str = DEFAULT_CALLBACK_URL,
) -> dict[str, Any]:
    """Exchange an authorization code for access + refresh tokens.

    Validates the state parameter, sends the token request with the PKCE
    verifier, and persists the tokens encrypted at rest.

    Returns the stored credential dict (without sensitive tokens exposed).
    """
    # Validate state
    if not _pending_flow.state or state != _pending_flow.state:
        raise ValueError("Invalid OAuth state parameter — possible CSRF attack.")

    verifier = _pending_flow.verifier
    if not verifier:
        raise ValueError("No pending OAuth flow — call build_authorize_url first.")

    # Exchange code for tokens
    payload = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CLIENT_ID,
        "code": code,
        "redirect_uri": callback_url,
        "code_verifier": verifier,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OPENAI_TOKEN_URL, data=payload)
        resp.raise_for_status()
        token_data = resp.json()

    # Extract and store credentials
    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_in = token_data.get("expires_in", 3600)
    scope = token_data.get("scope", "")

    # Extract account ID from the access token (JWT payload)
    account_id = _extract_account_id(access_token)

    credentials = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
        "scope": scope,
        "account_id": account_id,
        "created_at": time.time(),
    }

    _save_oauth_store(credentials)
    _pending_flow.clear()

    logger.info("ChatGPT OAuth credentials stored successfully (account: %s)", account_id)

    return {
        "status": "authorized",
        "account_id": account_id,
        "scope": scope,
        "expires_at": credentials["expires_at"],
    }


def _extract_account_id(access_token: str) -> str:
    """Extract the account ID from a JWT access token's payload.

    Returns an empty string if extraction fails (non-critical).
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return ""
        # JWT payload is base64url-encoded
        payload_b64 = parts[1]
        # Pad to multiple of 4
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub", payload.get("account_id", ""))
    except Exception:
        return ""


async def refresh_access_token() -> str | None:
    """Refresh the access token using the stored refresh token.

    Returns the new access token, or None if refresh fails.
    Updates the stored credentials on success.
    """
    store = _load_oauth_store()
    refresh_token = store.get("refresh_token")
    if not refresh_token:
        logger.warning("No refresh token available for ChatGPT OAuth.")
        return None

    payload = {
        "grant_type": "refresh_token",
        "client_id": OPENAI_CLIENT_ID,
        "refresh_token": refresh_token,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(OPENAI_TOKEN_URL, data=payload)
            resp.raise_for_status()
            token_data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("ChatGPT OAuth token refresh failed: %s", e)
        return None
    except Exception as e:
        logger.error("ChatGPT OAuth token refresh error: %s", e)
        return None

    new_access = token_data.get("access_token", "")
    new_refresh = token_data.get("refresh_token", refresh_token)
    expires_in = token_data.get("expires_in", 3600)

    store["access_token"] = new_access
    store["refresh_token"] = new_refresh
    store["expires_at"] = time.time() + expires_in
    _save_oauth_store(store)

    logger.info("ChatGPT OAuth access token refreshed successfully.")
    return new_access


async def get_access_token() -> str | None:
    """Get a valid access token, refreshing if needed.

    This is the main entry point for the LLM layer to get a token.
    Returns None if no OAuth credentials are stored or refresh fails.
    """
    store = _load_oauth_store()
    if not store.get("access_token"):
        return None

    # Check if token is expired (with 60s buffer)
    expires_at = store.get("expires_at", 0)
    if time.time() > (expires_at - 60):
        logger.info("ChatGPT OAuth token expired, refreshing...")
        return await refresh_access_token()

    return store["access_token"]


def get_oauth_status() -> dict[str, Any]:
    """Get the current OAuth authorization status.

    Returns a dict with status info (never exposes raw tokens).
    """
    store = _load_oauth_store()
    if not store.get("access_token"):
        return {
            "authorized": False,
            "account_id": None,
            "scope": None,
            "expires_at": None,
        }

    return {
        "authorized": True,
        "account_id": store.get("account_id", ""),
        "scope": store.get("scope", ""),
        "expires_at": store.get("expires_at"),
        "created_at": store.get("created_at"),
    }


def revoke_oauth() -> bool:
    """Revoke stored OAuth credentials.

    Deletes the local credential file. Returns True if credentials existed.
    """
    if _OAUTH_PATH.exists():
        _OAUTH_PATH.unlink()
        logger.info("ChatGPT OAuth credentials revoked.")
        return True
    return False


def is_oauth_configured() -> bool:
    """Check if ChatGPT OAuth credentials are stored and not empty."""
    store = _load_oauth_store()
    return bool(store.get("access_token"))
