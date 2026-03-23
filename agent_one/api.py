"""FastAPI backend — exposes Agent One as a REST + WebSocket API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent_one.app import AgentOneApp
from agent_one.core.config import update_config as _update_config
from agent_one.core.keystore import (
    save_key as _save_key,
    delete_key as _delete_key,
    delete_bedrock_keys as _delete_bedrock_keys,
    has_bedrock_keys as _has_bedrock_keys,
    list_providers as _list_providers,
    inject_keys_to_env,
    PROVIDER_ENV_VARS,
    BEDROCK_KEYS,
)
from agent_one.core.types import TraceStep

app = FastAPI(title="Agent One", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global app instance ---
_agent_app: AgentOneApp | None = None


def get_app() -> AgentOneApp:
    global _agent_app
    if _agent_app is None:
        _agent_app = AgentOneApp()
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


# --- REST endpoints ---

@app.on_event("startup")
async def startup():
    global _original_log
    # Inject any stored API keys into environment before first use
    inject_keys_to_env()
    agent_app = get_app()
    _original_log = agent_app.trace.log
    agent_app.trace.log = _patched_log


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent_app = get_app()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # Track trace steps for this request
    start_idx = len(agent_app.trace.steps)
    start = time.time()

    try:
        response = await agent_app.run(req.message)
    except Exception as e:
        response = f"Error: {type(e).__name__}: {e}"

    duration_ms = (time.time() - start) * 1000
    new_steps = agent_app.trace.steps[start_idx:]

    return ChatResponse(
        response=response,
        session_id=session_id,
        trace_steps=[s.model_dump() for s in new_steps],
        duration_ms=duration_ms,
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
    # Update the live app's config reference
    agent_app.config = new_config
    agent_app.router = __import__(
        "agent_one.routing.router", fromlist=["SmartRouter"]
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
        return {"status": "deleted" if deleted else "not_found", "provider": provider}

    deleted = _delete_key(provider)
    if deleted:
        # Remove from current environment too
        import os
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            os.environ.pop(env_var, None)
    return {"status": "deleted" if deleted else "not_found", "provider": provider}


@app.get("/api/models/{provider}")
async def list_models(provider: str):
    """Return available model names for a provider using litellm's model registry.

    Filters litellm's known model list by provider-specific prefixes or naming
    patterns. Only returns text/chat models (excludes image, audio, embedding).
    """
    from litellm import model_cost

    provider = provider.lower()

    # Provider prefix/pattern mapping.
    # Some providers use a "prefix/" style in litellm, others use name patterns.
    _PROVIDER_FILTERS: dict[str, dict] = {
        "openai": {"prefixes": ["gpt-", "o1", "o3", "o4", "chatgpt-"], "litellm_prefix": "openai/"},
        "anthropic": {"prefixes": ["claude-"]},
        "google": {"litellm_prefix": "gemini/"},
        "mistral": {"litellm_prefix": "mistral/"},
        "cohere": {"prefixes": ["command-"]},
        "deepseek": {"prefixes": ["deepseek-"], "litellm_prefix": "deepseek/"},
        "groq": {"litellm_prefix": "groq/"},
        "ollama": {"litellm_prefix": "ollama/"},
        "bedrock": {"litellm_prefix": "bedrock/"},
    }

    filt = _PROVIDER_FILTERS.get(provider)
    if not filt:
        return {"provider": provider, "models": []}

    # Patterns to exclude (images, audio, embedding, etc.)
    _EXCLUDE = (
        "dall-e", "tts-", "whisper", "embed", "moderation",
        "stable-diffusion", "canvas", "image", "video", "audio",
        "rerank", "polly", "transcribe",
    )

    models: set[str] = set()
    for key in model_cost:
        key_lower = key.lower()
        # Skip non-text models
        if any(ex in key_lower for ex in _EXCLUDE):
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
            # e.g. "bedrock/us-east-1/anthropic.claude-..." → "bedrock/anthropic.claude-..."
            if provider == "bedrock":
                parts = key.split("/")
                model_id = parts[-1]  # last component is always the model ID
                models.add(f"bedrock/{model_id}")
            else:
                models.add(key)

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
