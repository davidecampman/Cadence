"""Secure API key storage with Fernet symmetric encryption.

Keys are encrypted at rest using a machine-local master key.  The master key
is generated once and stored in the data directory.  Encrypted API keys are
persisted alongside it.

This module is intentionally self-contained — no external config dependency —
so it can be imported early in the startup sequence.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_DATA_DIR = Path(os.environ.get("CADENCE_DATA_DIR", "./data"))
_MASTER_KEY_PATH = _DATA_DIR / ".keystore_key"
_KEYS_PATH = _DATA_DIR / "api_keys.enc"

# Provider name → environment variable read by the SDKs / chat_completion()
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # AWS Bedrock — supports API key or IAM credentials (access key + secret)
    "bedrock_api_key": "BEDROCK_API_KEY",
    "bedrock_access_key_id": "AWS_ACCESS_KEY_ID",
    "bedrock_secret_access_key": "AWS_SECRET_ACCESS_KEY",
}

# Sub-keys that belong to the "bedrock" provider group
BEDROCK_KEYS = {"bedrock_api_key", "bedrock_access_key_id", "bedrock_secret_access_key"}


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_or_create_master_key() -> bytes:
    """Return the Fernet master key, creating it on first use."""
    _ensure_data_dir()
    if _MASTER_KEY_PATH.exists():
        return _MASTER_KEY_PATH.read_bytes().strip()
    key = Fernet.generate_key()
    _MASTER_KEY_PATH.write_bytes(key)
    # Restrict permissions to owner only
    try:
        os.chmod(_MASTER_KEY_PATH, 0o600)
    except OSError:
        pass  # Windows or restricted FS
    return key


def _fernet() -> Fernet:
    return Fernet(_get_or_create_master_key())


def _load_store() -> dict[str, str]:
    """Load and decrypt the key store. Returns {provider: api_key}."""
    if not _KEYS_PATH.exists():
        return {}
    try:
        cipher = _fernet()
        encrypted = _KEYS_PATH.read_bytes()
        decrypted = cipher.decrypt(encrypted)
        return json.loads(decrypted)
    except Exception as e:
        logger.warning("Failed to decrypt key store (keys may be lost if master key changed): %s", e)
        return {}


def _save_store(store: dict[str, str]) -> None:
    """Encrypt and persist the key store."""
    _ensure_data_dir()
    cipher = _fernet()
    plaintext = json.dumps(store).encode()
    _KEYS_PATH.write_bytes(cipher.encrypt(plaintext))
    try:
        os.chmod(_KEYS_PATH, 0o600)
    except OSError:
        pass


def save_key(provider: str, api_key: str) -> None:
    """Store an API key for a provider (encrypted at rest)."""
    store = _load_store()
    store[provider] = api_key
    _save_store(store)


def get_key(provider: str) -> str | None:
    """Retrieve a stored API key for a provider, or None."""
    return _load_store().get(provider)


def delete_key(provider: str) -> bool:
    """Delete a stored API key. Returns True if it existed."""
    store = _load_store()
    if provider not in store:
        return False
    del store[provider]
    _save_store(store)
    return True


def list_providers() -> list[str]:
    """Return provider names that have a stored key."""
    return list(_load_store().keys())


def has_bedrock_keys() -> bool:
    """Return True if any Bedrock credentials are stored."""
    store = _load_store()
    return any(k in store for k in BEDROCK_KEYS)


def delete_bedrock_keys() -> bool:
    """Delete all stored Bedrock credentials. Returns True if any existed."""
    store = _load_store()
    deleted = False
    for k in BEDROCK_KEYS:
        if k in store:
            del store[k]
            deleted = True
    if deleted:
        _save_store(store)
    return deleted


def inject_keys_to_env() -> list[str]:
    """Set environment variables for all stored keys.

    Only sets vars that are not already defined in the environment
    (env vars take precedence over stored keys).

    Returns the list of provider names that were injected.
    """
    store = _load_store()
    injected: list[str] = []
    for provider, api_key in store.items():
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var and not os.environ.get(env_var):
            # Inject if env var is missing OR empty
            os.environ[env_var] = api_key
            injected.append(provider)
    return injected
