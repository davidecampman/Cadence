"""OpenAI Sentinel system — proof-of-work and chat-requirements.

The ChatGPT ``/backend-api/conversation`` endpoint is protected by an
anti-abuse system called "Sentinel" that requires:

1. A preflight POST to ``/backend-api/sentinel/chat-requirements`` to
   obtain a requirements token, a proof-of-work seed, and difficulty.
2. A proof-of-work token computed via SHA3-512 hashcash over a fake
   browser-fingerprint config array.
3. Various headers (``openai-sentinel-chat-requirements-token``,
   ``openai-sentinel-proof-token``, ``oai-device-id``, etc.).

The proof-of-work algorithm and config array format are reverse-engineered
from the ChatGPT web client and open-source implementations (gpt4free,
chat2api, Webscout).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Client version extracted from the ChatGPT web client build.
# This must be updated periodically to match the latest production build.
OAI_CLIENT_VERSION = "prod-87d214a845e216ad4b68081d4b5cf8a222452106"
OAI_CLIENT_BUILD_NUMBER = "5879739"

# Google Tag Manager script URL (included in the fingerprint array)
_GTM_SCRIPT = "https://www.googletagmanager.com/gtag/js?id=G-9SHBSK2D9J"

# Sentinel endpoint
SENTINEL_CHAT_REQUIREMENTS_URL = "https://chatgpt.com/backend-api/sentinel/chat-requirements"

# Conversation endpoint (note the /f/ prefix used by the web client)
CONVERSATION_URL = "https://chatgpt.com/backend-api/f/conversation"

# Proof-of-work token prefix (matches the ChatGPT web client output)
_POW_PREFIX = "gAAAAAB"

# ---------------------------------------------------------------------------
# Browser fingerprint config
# ---------------------------------------------------------------------------

_SCREEN_VALUES = [3008, 4010, 6000, 6560, 8000]
_SCREEN_MULTIPLIERS = [1, 2, 4]

_REACT_CONTAINER_NAMES = [
    "__reactContainer$2okdysb58t1",
    "__reactContainer$cfilawjnerp",
    "__reactContainer$9ne2dfo1i47",
    "__reactContainer$410nzwhan2a",
    "__reactContainer$bm5vykej7vi",
]

_FEATURE_DETECTS = [
    "indexedDB",
    "localStorage",
    "sessionStorage",
    "performance",
]

_GPU_STRINGS = [
    "gpu\u2212[object GPU]",
    "gpu\u2212[object GPU]",
]

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)


def _build_config(user_agent: str | None = None) -> list[Any]:
    """Build a 25-element browser fingerprint config array.

    The structure matches the decoded ``openai-sentinel-proof-token`` from
    the ChatGPT web client (see field-by-field comments below).
    """
    ua = user_agent or _USER_AGENT
    screen = random.choice(_SCREEN_VALUES) * random.choice(_SCREEN_MULTIPLIERS)
    now = datetime.now(timezone.utc)
    # JavaScript-style date string (the exact format doesn't matter much,
    # but we match the pattern the server expects).
    date_str = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

    return [
        screen,                                 # [0]  screen width * multiplier
        date_str,                               # [1]  date string
        None,                                   # [2]  (stack hash — null is fine)
        0,                                      # [3]  iteration counter (set during PoW)
        ua,                                     # [4]  user agent
        _GTM_SCRIPT,                            # [5]  script URL
        OAI_CLIENT_VERSION,                     # [6]  oai-client-version
        "en-US",                                # [7]  navigator.language
        "en-US,en",                             # [8]  navigator.languages
        2,                                      # [9]  screen.colorDepth (÷ 12)
        random.choice(_GPU_STRINGS),            # [10] GPU info
        random.choice(_REACT_CONTAINER_NAMES),  # [11] React container prop
        random.choice(_FEATURE_DETECTS),        # [12] feature detection
        random.randint(10000, 999999),          # [13] random seed/hash
        str(uuid.uuid4()),                      # [14] UUID
        "",                                     # [15] (empty)
        16,                                     # [16] navigator.hardwareConcurrency
        int(time.time() * 1000),                # [17] timestamp in ms
        0, 0, 0, 0, 0,                          # [18-22] performance counters
        1,                                      # [23] flag
        1,                                      # [24] flag
    ]


# ---------------------------------------------------------------------------
# Proof-of-work solver
# ---------------------------------------------------------------------------

def generate_proof_token(
    seed: str,
    difficulty: str,
    user_agent: str | None = None,
) -> str:
    """Generate a proof-of-work token using SHA3-512 hashcash.

    The algorithm:
    1. Build a 25-element browser config array.
    2. For each iteration 0..100_000:
       - Set config[3] = iteration
       - JSON-encode → base64-encode the array
       - Compute SHA3-512(seed + base64_str)
       - If the hex digest starts with chars ≤ difficulty, return the token.
    3. If no solution found, return a fallback token.

    Args:
        seed: The seed string from the chat-requirements response.
        difficulty: The difficulty hex prefix from the chat-requirements response.
        user_agent: Optional user-agent string to embed in the config.

    Returns:
        A string like ``gAAAAAB<base64_data>`` that goes into the
        ``openai-sentinel-proof-token`` header.
    """
    config = _build_config(user_agent)
    diff_len = len(difficulty)

    for i in range(100_000):
        config[3] = i
        json_data = json.dumps(config, separators=(",", ":"))
        b64 = base64.b64encode(json_data.encode()).decode()
        hash_val = hashlib.sha3_512((seed + b64).encode()).digest()

        if hash_val.hex()[:diff_len] <= difficulty:
            return _POW_PREFIX + b64

    # Fallback — return a token even if we couldn't solve the PoW
    # (the server may still accept it, or it will 403 and we fall back to Codex)
    fallback_b64 = base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()
    return _POW_PREFIX + fallback_b64


def generate_fake_proof_token(user_agent: str | None = None) -> str:
    """Generate a plausible-looking proof token without actually solving PoW.

    Used as a fallback when we can't reach the chat-requirements endpoint.
    """
    config = _build_config(user_agent)
    config[3] = random.randint(0, 100)
    b64 = base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()
    return _POW_PREFIX + b64


# ---------------------------------------------------------------------------
# Chat-requirements preflight
# ---------------------------------------------------------------------------

async def fetch_chat_requirements(
    oauth_token: str,
    device_id: str,
    chatgpt_account_id: str = "",
) -> dict[str, Any]:
    """POST to ``/backend-api/sentinel/chat-requirements`` to get:

    - ``token``: the requirements token for the conversation request header.
    - ``proofofwork.seed``: the PoW seed.
    - ``proofofwork.difficulty``: the PoW difficulty.
    - ``proofofwork.required``: whether PoW is needed.

    Uses ``curl_cffi`` for TLS fingerprint impersonation, same as the
    conversation request itself.

    Returns the parsed JSON response dict, or an empty dict on failure.
    """
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": _USER_AGENT,
        "oai-device-id": device_id,
        "oai-language": "en-US",
        "oai-client-version": OAI_CLIENT_VERSION,
        "Referer": "https://chatgpt.com/",
        "Origin": "https://chatgpt.com",
    }
    if chatgpt_account_id:
        headers["ChatGPT-Account-Id"] = chatgpt_account_id

    try:
        from curl_cffi.requests import AsyncSession as CurlAsyncSession
    except ImportError:
        logger.warning("curl_cffi not available for sentinel preflight.")
        return {}

    try:
        async with CurlAsyncSession(impersonate="chrome") as session:
            resp = await session.post(
                SENTINEL_CHAT_REQUIREMENTS_URL,
                json={"p": None},
                headers=headers,
                timeout=30,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Sentinel chat-requirements returned %d: %s",
                    resp.status_code, resp.text[:300],
                )
                return {}
            data = resp.json()
            logger.info(
                "Sentinel chat-requirements: proofofwork.required=%s, "
                "proofofwork.difficulty=%s",
                data.get("proofofwork", {}).get("required"),
                data.get("proofofwork", {}).get("difficulty", "")[:8] + "...",
            )
            return data
    except Exception as e:
        logger.warning("Sentinel chat-requirements failed: %s", e)
        return {}


async def get_sentinel_headers(
    oauth_token: str,
    device_id: str,
    chatgpt_account_id: str = "",
    user_agent: str | None = None,
) -> dict[str, str]:
    """Get all sentinel-related headers needed for a conversation request.

    This is the main entry point — it:
    1. Calls the chat-requirements endpoint.
    2. Solves the proof-of-work if required.
    3. Returns a dict of extra headers to merge into the conversation request.
    """
    ua = user_agent or _USER_AGENT
    result: dict[str, str] = {}

    requirements = await fetch_chat_requirements(
        oauth_token, device_id, chatgpt_account_id,
    )

    # Chat-requirements token
    req_token = requirements.get("token", "")
    if req_token:
        result["openai-sentinel-chat-requirements-token"] = req_token

    # Proof-of-work
    pow_data = requirements.get("proofofwork", {})
    if pow_data.get("required"):
        seed = pow_data.get("seed", "")
        difficulty = pow_data.get("difficulty", "")
        if seed and difficulty:
            # Run CPU-bound PoW solver in a thread pool so the event loop stays free
            loop = asyncio.get_running_loop()
            proof = await loop.run_in_executor(
                None, generate_proof_token, seed, difficulty, ua
            )
            result["openai-sentinel-proof-token"] = proof
            logger.info("Sentinel PoW solved (difficulty=%s...)", difficulty[:8])
        else:
            result["openai-sentinel-proof-token"] = generate_fake_proof_token(ua)
    elif not requirements:
        # Couldn't reach requirements endpoint — send a fake proof anyway
        result["openai-sentinel-proof-token"] = generate_fake_proof_token(ua)

    return result
