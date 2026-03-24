"""FastAPI backend — exposes Sentinel as a REST + WebSocket API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sentinel.app import SentinelApp
from sentinel.core.config import update_config as _update_config
from sentinel.core.keystore import (
    save_key as _save_key,
    delete_key as _delete_key,
    delete_bedrock_keys as _delete_bedrock_keys,
    has_bedrock_keys as _has_bedrock_keys,
    list_providers as _list_providers,
    inject_keys_to_env,
    PROVIDER_ENV_VARS,
    BEDROCK_KEYS,
)
from sentinel.core.types import TraceStep

app = FastAPI(title="Sentinel", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global app instance ---
_agent_app: SentinelApp | None = None


def get_app() -> SentinelApp:
    global _agent_app
    if _agent_app is None:
        _agent_app = SentinelApp()
        _agent_app.discover_skills()
        # Disable console printing for API mode
        _agent_app.trace._console = False
    return _agent_app


# --- WebSocket connections for live trace streaming ---
_ws_clients: set[WebSocket] = set()


async def _broadcast_trace(step: TraceStep) -> None:
    """Send a trace step to all connected WebSocket clients."""
    global _ws_clients
    data = step.model_dump()
    data["timestamp"] = step.timestamp
    msg = json.dumps({"type": "trace", "data": data})
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


# Monkey-patch the trace logger to broadcast steps
_original_log = None


def _patched_log(step: TraceStep) -> None:
    _original_log(step)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_trace(step))
    except RuntimeError:
        pass


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_steps: list[dict]
    duration_ms: float
    context_turns: int = 0        # Current conversation turns in context
    max_context_turns: int = 50   # Max turns before hard cap


# --- Per-session conversation history ---
# Stores (role, content) tuples keyed by session_id so that follow-up
# messages carry prior context to the LLM.
_session_history: dict[str, list[dict[str, str]]] = {}

# Per-session compressed summary of older conversation turns.
# When compression fires, older turns are replaced by a single summary.
_session_summaries: dict[str, str] = {}


# --- REST endpoints ---

@app.on_event("startup")
async def startup():
    global _original_log
    # Inject any stored API keys into environment before first use
    inject_keys_to_env()
    agent_app = get_app()
    _original_log = agent_app.trace.log
    agent_app.trace.log = _patched_log


async def _compress_history(
    agent_app: SentinelApp,
    session_id: str,
    history: list[dict[str, str]],
    keep_recent: int = 10,
) -> list[dict[str, str]]:
    """Summarize older conversation turns into a condensed recap.

    Keeps the most recent ``keep_recent`` turn-pairs verbatim and compresses
    everything before that into a single summary message.
    """
    from sentinel.core.llm import chat_completion
    from sentinel.core.types import Message, Role

    # Split into old (to compress) and recent (to keep verbatim)
    split_idx = len(history) - (keep_recent * 2)
    if split_idx <= 0:
        return history

    old_turns = history[:split_idx]
    recent_turns = history[split_idx:]

    # Build the old conversation text
    old_text_parts = []
    for entry in old_turns:
        role_label = "User" if entry["role"] == "user" else "Assistant"
        old_text_parts.append(f"{role_label}: {entry['content']}")
    old_text = "\n".join(old_text_parts)

    # Prepend any existing summary so we don't lose earlier context
    existing_summary = _session_summaries.get(session_id, "")
    context_to_compress = ""
    if existing_summary:
        context_to_compress += f"[Previous summary]: {existing_summary}\n\n"
    context_to_compress += old_text

    compression_prompt = (
        "Summarize the following conversation into a concise recap. "
        "Preserve key facts, decisions, code/file references, and any "
        "commitments made. Be thorough but compact.\n\n"
        f"{context_to_compress}"
    )

    try:
        summary, _, _ = await chat_completion(
            model=agent_app.config.models.fast,
            messages=[Message(role=Role.USER, content=compression_prompt)],
            temperature=0.2,
            max_tokens=1024,
            bedrock_config=(
                agent_app.config.models.bedrock
                if agent_app.config.models.bedrock.enabled
                else None
            ),
        )
        _session_summaries[session_id] = summary.strip()
    except Exception:
        # If compression fails, fall back to hard truncation
        _session_summaries[session_id] = existing_summary
        return recent_turns

    # Return summary as a system-level context entry + recent verbatim turns
    compressed = [
        {"role": "assistant", "content": f"[Conversation summary]: {_session_summaries[session_id]}"},
    ]
    compressed.extend(recent_turns)
    return compressed


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent_app = get_app()
    conv_config = agent_app.config.conversation
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # Retrieve prior conversation history for this session
    history = list(_session_history.get(session_id, []))

    # Track trace steps for this request
    start_idx = len(agent_app.trace.steps)
    start = time.time()

    try:
        response = await agent_app.run(req.message, conversation_history=history)
    except Exception as e:
        err_str = str(e)
        if "AuthenticationError" in type(e).__name__ or "Missing" in err_str and "API Key" in err_str:
            # Provide a helpful hint about configuring API keys
            model_strong = agent_app.config.models.strong
            model_fast = agent_app.config.models.fast
            response = (
                f"Error: {type(e).__name__}: {e}\n\n"
                f"Hint: Your current models are strong={model_strong}, fast={model_fast}. "
                f"Please ensure you have saved the correct API key for your chosen provider "
                f"in the Config page, or change the model tier to a provider you have configured."
            )
        else:
            response = f"Error: {type(e).__name__}: {e}"

    # Persist this exchange in session history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": response})

    # Compress if enabled and threshold exceeded
    turn_count = len(history) // 2
    if conv_config.compression_enabled and turn_count > conv_config.compression_threshold:
        history = await _compress_history(agent_app, session_id, history)

    # Hard cap: trim to max_history_turns
    max_entries = conv_config.max_history_turns * 2
    if len(history) > max_entries:
        history = history[-max_entries:]

    _session_history[session_id] = history

    duration_ms = (time.time() - start) * 1000
    new_steps = agent_app.trace.steps[start_idx:]

    return ChatResponse(
        response=response,
        session_id=session_id,
        trace_steps=[s.model_dump() for s in new_steps],
        duration_ms=duration_ms,
        context_turns=len(_session_history[session_id]) // 2,
        max_context_turns=conv_config.max_history_turns,
    )


@app.get("/api/config")
async def get_config():
    return get_app().config.model_dump()


class ConfigUpdateRequest(BaseModel):
    updates: dict


@app.put("/api/config")
async def put_config(req: ConfigUpdateRequest):
    """Update configuration (partial merge). Returns the full updated config."""
    agent_app = get_app()
    new_config = _update_config(req.updates)
    # Update the live app's config reference AND propagate to sub-components
    agent_app.config = new_config
    agent_app.orchestrator.config = new_config
    agent_app.router = __import__(
        "sentinel.routing.router", fromlist=["SmartRouter"]
    ).SmartRouter(new_config)
    return new_config.model_dump()


@app.get("/api/skills")
async def get_skills():
    agent_app = get_app()
    skills = agent_app.skills.all_skills
    return [
        {"name": name, "version": s.version, "description": s.description}
        for name, s in skills.items()
    ]


@app.post("/api/skills/upload")
async def upload_skill(file: UploadFile):
    """Upload a skill zip file to install a new skill (or update an existing one)."""
    agent_app = get_app()
    if not file.filename or not file.filename.endswith(".zip"):
        return {"error": "File must be a .zip archive"}, 400
    data = await file.read()
    try:
        skill = agent_app.skills.install_from_zip(data)
    except ValueError as e:
        return {"error": str(e)}, 400
    return {
        "status": "installed",
        "skill": {"name": skill.name, "version": skill.version, "description": skill.description},
    }


@app.delete("/api/skills/{skill_name}")
async def delete_skill(skill_name: str):
    """Uninstall a skill by name, removing it from disk and memory."""
    agent_app = get_app()
    removed = agent_app.skills.uninstall(skill_name)
    if not removed:
        return {"error": f"Skill '{skill_name}' not found"}, 404
    return {"status": "uninstalled", "skill": skill_name}


@app.get("/api/tools")
async def get_tools():
    agent_app = get_app()
    defs = agent_app.tools.definitions()
    return [
        {"name": d.name, "description": d.description, "permission_tier": d.permission_tier}
        for d in defs
    ]


@app.get("/api/trace")
async def get_trace(limit: int = 50):
    agent_app = get_app()
    steps = agent_app.trace.steps[-limit:]
    return [s.model_dump() for s in steps]


class ApiKeyRequest(BaseModel):
    provider: str
    api_key: str


class BedrockKeysRequest(BaseModel):
    auth_type: str  # "api_key" or "iam"
    api_key: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None


@app.get("/api/keys")
async def get_keys():
    """Return which providers have stored keys (never exposes actual keys)."""
    stored = _list_providers()
    # Build providers dict, excluding bedrock sub-keys (they're grouped separately)
    providers = {
        name: {
            "env_var": env_var,
            "has_key": name in stored,
        }
        for name, env_var in PROVIDER_ENV_VARS.items()
        if name not in BEDROCK_KEYS
    }
    # Add bedrock as a grouped provider
    providers["bedrock"] = {
        "env_var": "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / BEDROCK_API_KEY",
        "has_key": _has_bedrock_keys(),
    }
    # For the stored list, collapse bedrock sub-keys into "bedrock"
    visible_stored = [s for s in stored if s not in BEDROCK_KEYS]
    if _has_bedrock_keys():
        visible_stored.append("bedrock")
    return {
        "providers": providers,
        "stored": visible_stored,
    }


@app.post("/api/keys")
async def post_key(req: ApiKeyRequest):
    """Store an API key for a provider (encrypted at rest)."""
    provider = req.provider.lower()
    if provider not in PROVIDER_ENV_VARS:
        return {"error": f"Unknown provider: {provider}"}, 400
    _save_key(provider, req.api_key)
    # Also inject into the current process so it takes effect immediately
    inject_keys_to_env()
    # Force-set even if env was already defined (user explicitly updated)
    import os
    os.environ[PROVIDER_ENV_VARS[provider]] = req.api_key
    return {"status": "saved", "provider": provider}


@app.post("/api/keys/bedrock")
async def post_bedrock_keys(req: BedrockKeysRequest):
    """Store AWS Bedrock credentials (encrypted at rest).

    Supports two auth types:
    - "api_key": A long-term Bedrock API key
    - "iam": AWS access key ID + secret access key
    """
    import os

    # Clear any existing bedrock keys first so we don't mix auth types
    _delete_bedrock_keys()
    for k in BEDROCK_KEYS:
        env_var = PROVIDER_ENV_VARS.get(k)
        if env_var:
            os.environ.pop(env_var, None)

    if req.auth_type == "api_key":
        if not req.api_key:
            return {"error": "api_key is required for auth_type 'api_key'"}, 400
        _save_key("bedrock_api_key", req.api_key)
        os.environ["BEDROCK_API_KEY"] = req.api_key
    elif req.auth_type == "iam":
        if not req.access_key_id or not req.secret_access_key:
            return {"error": "access_key_id and secret_access_key are required for auth_type 'iam'"}, 400
        _save_key("bedrock_access_key_id", req.access_key_id)
        _save_key("bedrock_secret_access_key", req.secret_access_key)
        os.environ["AWS_ACCESS_KEY_ID"] = req.access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = req.secret_access_key
    else:
        return {"error": f"Unknown auth_type: {req.auth_type}"}, 400

    inject_keys_to_env()

    # Auto-enable bedrock in config when credentials are saved
    agent_app = get_app()
    if not agent_app.config.models.bedrock.enabled:
        new_config = _update_config({"models": {"bedrock": {"enabled": True}}})
        agent_app.config = new_config
        agent_app.orchestrator.config = new_config

    return {"status": "saved", "provider": "bedrock", "auth_type": req.auth_type}


@app.delete("/api/keys/{provider}")
async def remove_key(provider: str):
    """Delete a stored API key."""
    provider = provider.lower()
    # Handle bedrock as a group
    if provider == "bedrock":
        deleted = _delete_bedrock_keys()
        if deleted:
            import os
            for k in BEDROCK_KEYS:
                env_var = PROVIDER_ENV_VARS.get(k)
                if env_var:
                    os.environ.pop(env_var, None)
            # Auto-disable bedrock in config when credentials are removed
            agent_app = get_app()
            if agent_app.config.models.bedrock.enabled:
                new_config = _update_config({"models": {"bedrock": {"enabled": False}}})
                agent_app.config = new_config
                agent_app.orchestrator.config = new_config
        return {"status": "deleted" if deleted else "not_found", "provider": provider}

    deleted = _delete_key(provider)
    if deleted:
        # Remove from current environment too
        import os
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            os.environ.pop(env_var, None)
    return {"status": "deleted" if deleted else "not_found", "provider": provider}


# --- OpenRouter dynamic model discovery ---
# OpenRouter exposes two public endpoints:
#   /api/v1/models            – chat / completion models
#   /api/v1/embeddings/models – embedding models
# We query both and cache results so the UI always reflects what's actually
# available, rather than relying on litellm's static registry.

_openrouter_cache: dict[str, list[str]] = {}  # "chat" | "embedding" -> model ids
_openrouter_cache_ts: float = 0.0
_OPENROUTER_CACHE_TTL = 3600  # re-fetch every hour

logger = logging.getLogger(__name__)


async def _fetch_openrouter_models() -> dict[str, list[str]]:
    """Fetch chat and embedding models from OpenRouter, with caching.

    Returns a dict with keys ``"chat"`` and ``"embedding"``, each mapping to
    a list of litellm-style model ids (``openrouter/<provider>/<model>``).
    Falls back to ``None`` (caller should use litellm registry) on failure.
    """
    global _openrouter_cache, _openrouter_cache_ts

    if _openrouter_cache and (time.time() - _openrouter_cache_ts) < _OPENROUTER_CACHE_TTL:
        return _openrouter_cache

    import httpx

    result: dict[str, list[str]] = {"chat": [], "embedding": []}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            chat_resp, embed_resp = await asyncio.gather(
                client.get("https://openrouter.ai/api/v1/models"),
                client.get("https://openrouter.ai/api/v1/embeddings/models"),
                return_exceptions=True,
            )

            if isinstance(chat_resp, BaseException):
                logger.debug("OpenRouter /models fetch failed: %s", chat_resp)
            else:
                chat_resp.raise_for_status()
                body = chat_resp.json()
                data = body.get("data", body if isinstance(body, list) else [])
                result["chat"] = [f"openrouter/{m['id']}" for m in data if "id" in m]

            if isinstance(embed_resp, BaseException):
                logger.debug("OpenRouter /embeddings/models fetch failed: %s", embed_resp)
            else:
                embed_resp.raise_for_status()
                body = embed_resp.json()
                data = body.get("data", body if isinstance(body, list) else [])
                result["embedding"] = [f"openrouter/{m['id']}" for m in data if "id" in m]

        if result["chat"] or result["embedding"]:
            _openrouter_cache = result
            _openrouter_cache_ts = time.time()
            return result
    except Exception as exc:
        logger.debug("Failed to fetch OpenRouter models: %s", exc)

    # Return empty – caller will fall back to litellm registry
    return result


@app.get("/api/models/{provider}")
async def list_models(provider: str, tier: str | None = None):
    """Return available model names for a provider using litellm's model registry.

    Filters litellm's known model list by provider-specific prefixes or naming
    patterns. When ``tier`` is provided (strong, fast, embedding), further
    filters to models appropriate for that use case.
    """
    from litellm import model_cost

    provider = provider.lower()
    tier = tier.lower() if tier else None

    # Provider prefix/pattern mapping.
    # Some providers use a "prefix/" style in litellm, others use name patterns.
    _PROVIDER_FILTERS: dict[str, dict] = {
        "openai": {"prefixes": ["gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding-"], "litellm_prefix": "openai/"},
        "anthropic": {"prefixes": ["claude-", "voyage-"]},
        "google": {"litellm_prefix": "gemini/"},
        "mistral": {"prefixes": ["mistral-embed"], "litellm_prefix": "mistral/"},
        "cohere": {"prefixes": ["command-", "embed-"]},
        "deepseek": {"prefixes": ["deepseek-"], "litellm_prefix": "deepseek/"},
        "groq": {"litellm_prefix": "groq/"},
        "openrouter": {"litellm_prefix": "openrouter/"},
        "ollama": {"litellm_prefix": "ollama/"},
        "bedrock": {"litellm_prefix": "bedrock/"},
    }

    filt = _PROVIDER_FILTERS.get(provider)
    if not filt:
        return {"provider": provider, "models": []}

    # --- OpenRouter: use their live API for all tiers ---
    if provider == "openrouter":
        or_models = await _fetch_openrouter_models()
        models: set[str] = set()
        if tier == "embedding":
            models.update(or_models.get("embedding", []))
        elif tier in ("strong", "fast"):
            models.update(or_models.get("chat", []))
        else:
            # No tier specified: show chat models (exclude embeddings)
            models.update(or_models.get("chat", []))

        # If the live API returned results, use them; otherwise fall through
        # to the litellm registry below as a last resort.
        if models:
            return {"provider": provider, "models": sorted(models)}

    # --- Tier-based exclusion / inclusion patterns ---
    # Embedding tier: only show embedding models.
    # Strong/Fast tiers: exclude embedding, image, audio, etc.
    # No tier: show all text/chat models (existing behaviour).

    _EXCLUDE_NON_TEXT = (
        "dall-e", "tts-", "whisper", "moderation",
        "stable-diffusion", "canvas", "image", "video", "audio",
        "rerank", "polly", "transcribe",
    )

    _EMBEDDING_KEYWORDS = ("embed",)

    models: set[str] = set()
    for key in model_cost:
        key_lower = key.lower()

        # Always skip non-text media models
        if any(ex in key_lower for ex in _EXCLUDE_NON_TEXT):
            continue

        is_embedding = any(kw in key_lower for kw in _EMBEDDING_KEYWORDS)

        # Tier-based filtering
        if tier == "embedding" and not is_embedding:
            continue
        if tier in ("strong", "fast") and is_embedding:
            continue
        # When no tier specified, exclude embeddings (original behaviour)
        if tier is None and is_embedding:
            continue

        matched = False
        # Check litellm prefix (e.g. "bedrock/...", "gemini/...")
        if filt.get("litellm_prefix") and key_lower.startswith(filt["litellm_prefix"]):
            matched = True
        # Check name prefixes (e.g. "gpt-", "claude-")
        for pfx in filt.get("prefixes", []):
            if key_lower.startswith(pfx):
                matched = True
                break

        if matched:
            # For bedrock, deduplicate regional/commitment variants
            # e.g. "bedrock/converse/us-east-1/anthropic.claude-..." → "bedrock/converse/anthropic.claude-..."
            # Always use the converse route as it's the recommended Bedrock API path.
            if provider == "bedrock":
                parts = key.split("/")
                model_id = parts[-1]  # last component is always the model ID
                models.add(f"bedrock/converse/{model_id}")
            else:
                models.add(key)

    # For bedrock, also include models from our explicit mapping so users
    # always see the latest Claude models even if litellm's registry is outdated.
    if provider == "bedrock":
        from sentinel.core.llm import _BEDROCK_MODEL_MAP, _region_to_inference_prefix
        _bedrock_region = get_app().config.models.bedrock.region
        _inf_prefix = _region_to_inference_prefix(_bedrock_region)
        for _std_name, _br_id in _BEDROCK_MODEL_MAP.items():
            candidate = f"bedrock/converse/{_inf_prefix}.{_br_id}"
            is_embedding = any(kw in _br_id.lower() for kw in _EMBEDDING_KEYWORDS)
            if tier == "embedding" and not is_embedding:
                continue
            if tier in ("strong", "fast") and is_embedding:
                continue
            if tier is None and is_embedding:
                continue
            models.add(candidate)

    return {"provider": provider, "models": sorted(models)}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# --- WebSocket for live trace streaming ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            # Keep connection alive; client can send pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _ws_clients.discard(ws)


# --- Serve frontend static files ---

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        file_path = _frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_frontend_dist / "index.html")
