"""FastAPI backend — exposes Cadence as a REST + WebSocket API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Ensure stdout/stderr use UTF-8 so emoji in agent output / trace logs don't
# crash the process on systems where the default codec is 'charmap'.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import platform
import subprocess

from fastapi import FastAPI, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cadence.app import CadenceApp
from cadence.core.config import update_config as _update_config
from cadence.core.streaming import StreamCollector, StreamEvent
from cadence.core.keystore import (
    save_key as _save_key,
    delete_key as _delete_key,
    delete_bedrock_keys as _delete_bedrock_keys,
    has_bedrock_keys as _has_bedrock_keys,
    list_providers as _list_providers,
    inject_keys_to_env,
    PROVIDER_ENV_VARS,
    BEDROCK_KEYS,
)
from cadence.core.chatgpt_oauth import (
    build_authorize_url,
    exchange_code,
    get_oauth_status,
    revoke_oauth,
    ensure_callback_server,
    DEFAULT_CALLBACK_PORT,
    DEFAULT_CALLBACK_URL,
)
from cadence.core.types import TraceStep
from cadence.storage.chat_store import ChatStore, ChatMessageRecord

app = FastAPI(title="Cadence", version="0.1.0")

_default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_cors_origins = os.environ.get("CADENCE_CORS_ORIGINS", "").split(",") if os.environ.get("CADENCE_CORS_ORIGINS") else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_api_responses(request: Request, call_next):
    """Prevent browsers from caching API responses."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


# --- Global app instance ---
_agent_app: CadenceApp | None = None
_chat_store: ChatStore | None = None

# Maps session_id → running asyncio.Task so they can be cancelled
_running_tasks: dict[str, asyncio.Task] = {}

# Concurrency limiter for chat requests
_MAX_CONCURRENT_REQUESTS = 5
_chat_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

# Request timeout (seconds)
_CHAT_REQUEST_TIMEOUT = 600  # 10 minutes


def get_app() -> CadenceApp:
    global _agent_app
    if _agent_app is None:
        _agent_app = CadenceApp()
        _agent_app.discover_skills()
        # Disable console printing for API mode
        _agent_app.trace._console = False
    return _agent_app


def get_chat_store() -> ChatStore:
    global _chat_store
    if _chat_store is None:
        _chat_store = ChatStore()
    return _chat_store


# --- WebSocket connections with per-client async queues ---

class WSClient:
    """A WebSocket client with its own async message queue for backpressure handling."""

    __slots__ = ("ws", "queue", "consecutive_failures")

    def __init__(self, ws: WebSocket, max_queue: int = 100):
        self.ws = ws
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue)
        self.consecutive_failures = 0

    async def send(self, msg: str) -> bool:
        """Enqueue a message without blocking. Returns False if queue is full."""
        try:
            self.queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            self.consecutive_failures += 1
            return False

    async def drain(self) -> None:
        """Process queued messages. Runs as a background task per client."""
        while True:
            msg = await self.queue.get()
            if msg is None:  # Shutdown signal
                break
            try:
                await self.ws.send_text(msg)
                self.consecutive_failures = 0
            except Exception:
                self.consecutive_failures += 1
                if self.consecutive_failures > 5:
                    break


_ws_clients: dict[int, WSClient] = {}
_ws_drain_tasks: dict[int, asyncio.Task] = {}


async def _broadcast_trace(step: TraceStep) -> None:
    """Send a trace step to all connected WebSocket clients (non-blocking)."""
    data = step.model_dump()
    data["timestamp"] = step.timestamp
    msg = json.dumps({"type": "trace", "data": data})
    dead = []
    for cid, client in _ws_clients.items():
        ok = await client.send(msg)
        if not ok and client.consecutive_failures > 10:
            dead.append(cid)
    for cid in dead:
        _remove_ws_client(cid)


async def _broadcast_dag(dag_data: dict) -> None:
    """Send a DAG snapshot to all connected WebSocket clients (non-blocking)."""
    msg = json.dumps({"type": "dag_update", "data": dag_data})
    dead = []
    for cid, client in _ws_clients.items():
        ok = await client.send(msg)
        if not ok and client.consecutive_failures > 10:
            dead.append(cid)
    for cid in dead:
        _remove_ws_client(cid)


def _remove_ws_client(cid: int) -> None:
    """Clean up a WebSocket client."""
    client = _ws_clients.pop(cid, None)
    task = _ws_drain_tasks.pop(cid, None)
    if task and not task.done():
        task.cancel()
    if client:
        try:
            client.queue.put_nowait(None)  # Signal drain to stop
        except asyncio.QueueFull:
            pass


def _serialize_dag(dag) -> dict:
    """Convert a TaskDAG into a JSON-serialisable dict of nodes and edges."""
    from cadence.agents.orchestrator import TaskDAG
    nodes = []
    edges = []
    for task in dag._tasks.values():
        nodes.append({
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "role": task.metadata.get("role", "general"),
            "assigned_agent": task.assigned_agent,
        })
        for dep_id in task.dependencies:
            edges.append({"from": dep_id, "to": task.id})
    return {"nodes": nodes, "edges": edges}


# Monkey-patch the trace logger to broadcast steps
_original_log = None


def _patched_log(step: TraceStep) -> None:
    _original_log(step)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_trace(step))
    except RuntimeError:
        pass


def _on_task_update(dag) -> None:
    """Called by the orchestrator whenever a task changes status."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_dag(_serialize_dag(dag)))
    except RuntimeError:
        pass


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    images: list[dict] | None = None  # Optional image inputs: [{"data": "base64...", "media_type": "image/png"}]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_steps: list[dict]
    duration_ms: float
    context_turns: int = 0        # Current conversation turns in context
    max_context_turns: int = 50   # Max turns before hard cap


# --- Per-session conversation history ---
# Now backed by SQLite via ChatStore. In-memory dicts removed.


# --- REST endpoints ---

@app.on_event("startup")
async def startup():
    global _original_log
    # Inject any stored API keys into environment before first use
    inject_keys_to_env()
    agent_app = get_app()
    _original_log = agent_app.trace.log
    agent_app.trace.log = _patched_log
    # Wire DAG update callback for live graph streaming
    agent_app.orchestrator.on_task_update = _on_task_update
    # Connect to configured MCP servers
    await agent_app.connect_mcp_servers()


@app.on_event("shutdown")
async def shutdown():
    from cadence.tools.browser import _shutdown_browser
    await _shutdown_browser()
    # Disconnect MCP servers
    agent_app = get_app()
    await agent_app.disconnect_mcp_servers()


async def _compress_history(
    agent_app: CadenceApp,
    session_id: str,
    history: list[dict[str, str]],
    keep_recent: int = 10,
) -> list[dict[str, str]]:
    """Summarize older conversation turns into a condensed recap.

    Keeps the most recent ``keep_recent`` turn-pairs verbatim and compresses
    everything before that into a single summary message.
    """
    from cadence.core.llm import chat_completion
    from cadence.core.types import Message, Role

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
    store = get_chat_store()
    existing_summary = store.get_session_summary(session_id)
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
            local_config=(
                agent_app.config.models.local
                if agent_app.config.models.local.enabled
                else None
            ),
        )
        store.save_session_summary(session_id, summary.strip())
    except Exception:
        # If compression fails, fall back to hard truncation
        store.save_session_summary(session_id, existing_summary)
        return recent_turns

    # Return summary as a system-level context entry + recent verbatim turns
    final_summary = store.get_session_summary(session_id)
    compressed = [
        {"role": "assistant", "content": f"[Conversation summary]: {final_summary}"},
    ]
    compressed.extend(recent_turns)
    return compressed


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent_app = get_app()
    store = get_chat_store()
    conv_config = agent_app.config.conversation
    session_id = req.session_id or str(uuid.uuid4())[:8]

    # Concurrency limiting: queue requests when too many are in-flight
    async with _chat_semaphore:
        # Retrieve prior conversation history from persistent store
        history = list(store.get_session_history(session_id))

        # Track trace steps for this request
        start_idx = len(agent_app.trace.steps)
        start = time.time()

        try:
            # Wrap orchestrator call in a timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                agent_app.run(req.message, conversation_history=history, images=req.images),
                timeout=_CHAT_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            response = (
                f"Request timed out after {_CHAT_REQUEST_TIMEOUT}s. "
                "The request was too complex or the model took too long to respond. "
                "Try breaking it into smaller parts."
            )
        except Exception as e:
            err_str = str(e)
            if "AuthenticationError" in type(e).__name__ or ("Missing" in err_str and "API Key" in err_str):
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

        # Hard cap: trim to max_history_turns (with warning at 80%)
        max_entries = conv_config.max_history_turns * 2
        warn_entries = int(max_entries * 0.8)
        if len(history) > max_entries:
            history = history[-max_entries:]

        store.save_session_history(
            session_id, history,
            summary=store.get_session_summary(session_id),
        )

        duration_ms = (time.time() - start) * 1000
        new_steps = agent_app.trace.steps[start_idx:]

        result = ChatResponse(
            response=response,
            session_id=session_id,
            trace_steps=[s.model_dump() for s in new_steps],
            duration_ms=duration_ms,
            context_turns=len(history) // 2,
            max_context_turns=conv_config.max_history_turns,
        )

        # Append context warning if approaching the limit
        if len(history) >= warn_entries:
            result.response += (
                f"\n\n---\n*Context usage: {len(history) // 2}/{conv_config.max_history_turns} turns. "
                "Consider starting a new session to avoid context truncation.*"
            )

        return result


# --- Chat persistence CRUD endpoints ---

class CreateChatRequest(BaseModel):
    id: str | None = None
    title: str = "New Chat"
    created_at: float | None = None


class UpdateChatRequest(BaseModel):
    title: str | None = None
    session_id: str | None = None


class AddMessageRequest(BaseModel):
    id: str
    role: str
    content: str
    timestamp: float
    duration_ms: float | None = None
    trace_steps: list[dict] | None = None


@app.get("/api/chats")
async def list_chats():
    """List all chats (metadata only, no messages)."""
    store = get_chat_store()
    chats = store.list_chats()
    return [c.model_dump(exclude={"messages"}) for c in chats]


@app.get("/api/chats/{chat_id}")
async def get_single_chat(chat_id: str):
    """Get a chat with all its messages."""
    store = get_chat_store()
    chat = store.get_chat(chat_id)
    if not chat:
        return {"error": "Chat not found"}, 404
    return chat.model_dump()


@app.post("/api/chats")
async def create_chat(req: CreateChatRequest):
    """Create a new chat."""
    store = get_chat_store()
    chat = store.create_chat(
        chat_id=req.id, title=req.title, created_at=req.created_at
    )
    return chat.model_dump()


@app.put("/api/chats/{chat_id}")
async def update_single_chat(chat_id: str, req: UpdateChatRequest):
    """Update chat metadata."""
    store = get_chat_store()
    chat = store.update_chat(chat_id, title=req.title, session_id=req.session_id)
    if not chat:
        return {"error": "Chat not found"}, 404
    return chat.model_dump()


@app.delete("/api/chats/{chat_id}")
async def delete_single_chat(chat_id: str):
    """Delete a chat and all its messages."""
    store = get_chat_store()
    deleted = store.delete_chat(chat_id)
    return {"status": "deleted" if deleted else "not_found"}


@app.post("/api/chats/{chat_id}/messages")
async def add_chat_message(chat_id: str, req: AddMessageRequest):
    """Add a message to a chat."""
    store = get_chat_store()
    msg = ChatMessageRecord(
        id=req.id,
        chat_id=chat_id,
        role=req.role,
        content=req.content,
        timestamp=req.timestamp,
        duration_ms=req.duration_ms,
        trace_steps=req.trace_steps,
    )
    store.add_message(msg)
    return {"status": "ok"}


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
    if agent_app.orchestrator.prompt_evolver:
        agent_app.orchestrator.prompt_evolver.config = new_config
    agent_app.router = __import__(
        "cadence.routing.router", fromlist=["SmartRouter"]
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


_MAX_SKILL_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/api/skills/upload")
async def upload_skill(file: UploadFile):
    """Upload a skill zip file to install a new skill (or update an existing one)."""
    agent_app = get_app()
    if not file.filename or not file.filename.endswith(".zip"):
        return {"error": "File must be a .zip archive"}, 400
    # Read with size limit to prevent OOM on large uploads
    data = await file.read(_MAX_SKILL_UPLOAD_BYTES + 1)
    if len(data) > _MAX_SKILL_UPLOAD_BYTES:
        return {"error": f"File too large (max {_MAX_SKILL_UPLOAD_BYTES // 1024 // 1024} MB)"}, 400
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


@app.get("/api/mcp")
async def get_mcp_status():
    """Return connection status for all MCP servers."""
    agent_app = get_app()
    return agent_app.mcp_manager.status()


@app.get("/api/trace")
async def get_trace(limit: int = 50):
    agent_app = get_app()
    steps = agent_app.trace.steps[-limit:]
    return [s.model_dump() for s in steps]


@app.get("/api/dag")
async def get_dag():
    """Return the current task DAG as nodes + edges for graph visualisation."""
    agent_app = get_app()
    return _serialize_dag(agent_app.orchestrator.dag)


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
        if agent_app.orchestrator.prompt_evolver:
            agent_app.orchestrator.prompt_evolver.config = new_config

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
                if agent_app.orchestrator.prompt_evolver:
                    agent_app.orchestrator.prompt_evolver.config = new_config
        return {"status": "deleted" if deleted else "not_found", "provider": provider}

    deleted = _delete_key(provider)
    if deleted:
        # Remove from current environment too
        import os
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            os.environ.pop(env_var, None)
    return {"status": "deleted" if deleted else "not_found", "provider": provider}


# --- ChatGPT OAuth endpoints ---

class OAuthInitRequest(BaseModel):
    callback_url: str | None = None


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str
    callback_url: str | None = None


@app.post("/api/oauth/chatgpt/initiate")
async def oauth_chatgpt_initiate(req: OAuthInitRequest | None = None):
    """Start the ChatGPT OAuth PKCE flow.

    Spins up a temporary callback server on port 1455 (the only port OpenAI
    accepts for this client ID), then returns the authorization URL.
    """
    callback = (req.callback_url if req and req.callback_url else None) or DEFAULT_CALLBACK_URL
    await ensure_callback_server()
    auth_url = build_authorize_url(callback_url=callback)
    return {
        "authorize_url": auth_url,
        "callback_url": callback,
        "callback_port": DEFAULT_CALLBACK_PORT,
    }


@app.post("/api/oauth/chatgpt/callback")
async def oauth_chatgpt_callback(req: OAuthCallbackRequest):
    """Complete the OAuth flow by exchanging the authorization code for tokens.

    The frontend captures the code from the callback redirect and sends it here.
    """
    callback = req.callback_url or DEFAULT_CALLBACK_URL
    try:
        result = await exchange_code(
            code=req.code,
            state=req.state,
            callback_url=callback,
        )
        return result
    except ValueError as e:
        return {"error": str(e), "status": "failed"}
    except Exception as e:
        logger.error("ChatGPT OAuth callback failed: %s", e)
        return {"error": f"Token exchange failed: {e}", "status": "failed"}


@app.get("/api/oauth/chatgpt/status")
async def oauth_chatgpt_status():
    """Check the current ChatGPT OAuth authorization status."""
    status = get_oauth_status()
    # Include fallback info so the UI can show it
    status["has_api_key_fallback"] = bool(os.environ.get("OPENAI_API_KEY"))
    return status


@app.post("/api/oauth/chatgpt/revoke")
async def oauth_chatgpt_revoke():
    """Revoke ChatGPT OAuth credentials."""
    revoked = revoke_oauth()
    return {"status": "revoked" if revoked else "not_found"}


# --- OpenRouter dynamic model discovery ---
# OpenRouter exposes two public endpoints:
#   /api/v1/models            – chat / completion models
#   /api/v1/embeddings/models – embedding models
# We query both and cache results so the UI always reflects what's actually
# available.

_openrouter_cache: dict[str, list[str]] = {}  # "chat" | "embedding" -> model ids
_openrouter_cache_ts: float = 0.0
_OPENROUTER_CACHE_TTL = 3600  # re-fetch every hour

logger = logging.getLogger(__name__)


async def _fetch_openrouter_models() -> dict[str, list[str]]:
    """Fetch chat and embedding models from OpenRouter, with caching.

    Returns a dict with keys ``"chat"`` and ``"embedding"``, each mapping to
    a list of model ids (``openrouter/<provider>/<model>``).
    Returns empty lists on failure.
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

    # Return empty – caller will use static registry
    return result


# Static model registry of well-known models.  Only needs to list
# well-known model names so the /api/models endpoint can enumerate them.
_KNOWN_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
    ],
    "anthropic": [
        "claude-opus-4-6-20250610", "claude-sonnet-4-6-20250610",
        "claude-sonnet-4-5-20250514", "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    ],
    "google": [
        "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash",
    ],
    "mistral": [
        "mistral/mistral-large-latest", "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest", "mistral/codestral-latest",
        "mistral-embed",
    ],
    "cohere": [
        "command-r-plus", "command-r", "embed-english-v3.0", "embed-multilingual-v3.0",
    ],
    "deepseek": ["deepseek/deepseek-chat", "deepseek/deepseek-coder"],
    "groq": [
        "groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant",
        "groq/mixtral-8x7b-32768", "groq/gemma2-9b-it",
    ],
}

_EMBEDDING_KEYWORDS = ("embed",)


@app.get("/api/models/{provider}")
async def list_models(provider: str, tier: str | None = None):
    """Return available model names for a provider.

    Uses a static registry for most providers, the live OpenRouter API for
    OpenRouter, and ``_BEDROCK_MODEL_MAP`` for Bedrock.
    """
    provider = provider.lower()
    tier = tier.lower() if tier else None

    # --- OpenRouter: use their live API ---
    if provider == "openrouter":
        or_models = await _fetch_openrouter_models()
        models: set[str] = set()
        if tier == "embedding":
            models.update(or_models.get("embedding", []))
        else:
            models.update(or_models.get("chat", []))
        if models:
            return {"provider": provider, "models": sorted(models)}
        # Fallback: return empty rather than stale data
        return {"provider": provider, "models": []}

    # --- Bedrock: use our explicit model map ---
    if provider == "bedrock":
        from cadence.core.llm import _BEDROCK_MODEL_MAP, _region_to_inference_prefix
        _bedrock_region = get_app().config.models.bedrock.region
        _inf_prefix = _region_to_inference_prefix(_bedrock_region)
        models = set()
        for _std_name, _br_id in _BEDROCK_MODEL_MAP.items():
            is_embedding = any(kw in _br_id.lower() for kw in _EMBEDDING_KEYWORDS)
            if tier == "embedding" and not is_embedding:
                continue
            if tier in ("strong", "fast") and is_embedding:
                continue
            if tier is None and is_embedding:
                continue
            models.add(f"bedrock/converse/{_inf_prefix}.{_br_id}")
        return {"provider": provider, "models": sorted(models)}

    # --- Static registry for other providers ---
    known = _KNOWN_MODELS.get(provider)
    if not known:
        return {"provider": provider, "models": []}

    models = set()
    for key in known:
        is_embedding = any(kw in key.lower() for kw in _EMBEDDING_KEYWORDS)
        if tier == "embedding" and not is_embedding:
            continue
        if tier in ("strong", "fast") and is_embedding:
            continue
        if tier is None and is_embedding:
            continue
        models.add(key)

    return {"provider": provider, "models": sorted(models)}


# --- Local file access endpoints ---

# Directories that the file download/reveal endpoints are allowed to serve from.
_ALLOWED_FILE_ROOTS = [
    Path.cwd().resolve(),
    Path(tempfile.gettempdir()).resolve(),
]


def _is_path_allowed(p: Path) -> bool:
    """Return True if *p* falls under one of the allowed root directories."""
    resolved = p.resolve()
    return any(
        resolved == root or str(resolved).startswith(str(root) + os.sep)
        for root in _ALLOWED_FILE_ROOTS
    )


@app.get("/api/files/download")
async def download_file(path: str = Query(..., description="Absolute path to the file")):
    """Serve a local file as a browser download."""
    p = Path(path).expanduser().resolve()
    if not _is_path_allowed(p):
        return {"error": "Access denied: path outside allowed directories"}, 403
    if not p.exists():
        return {"error": f"File not found: {path}"}, 404
    if not p.is_file():
        return {"error": f"Not a file: {path}"}, 400
    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type="application/octet-stream",
    )


@app.get("/api/files/reveal")
async def reveal_file(path: str = Query(..., description="Path to reveal in file manager")):
    """Open the OS file manager to the directory containing the given file."""
    p = Path(path).expanduser().resolve()
    if not _is_path_allowed(p):
        return {"error": "Access denied: path outside allowed directories"}, 403
    target = p.parent if p.is_file() else p
    if not target.exists():
        return {"error": f"Directory not found: {target}"}

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(target)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return {"status": "opened", "path": str(target)}
    except FileNotFoundError:
        return {"error": f"No file manager found for {system}"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# --- Knowledge base endpoints ---

_kb_store = None


def get_kb_store():
    global _kb_store
    if _kb_store is None:
        from cadence.knowledge.store import KnowledgeStore
        _kb_store = KnowledgeStore()
    return _kb_store


class KBIngestRequest(BaseModel):
    source: str  # file path, URL, or raw text
    title: str
    source_type: str | None = None  # auto-detected if omitted


class KBSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    source_filter: str | None = None


@app.post("/api/kb/ingest")
async def kb_ingest(req: KBIngestRequest):
    """Ingest a document into the knowledge base."""
    from cadence.knowledge.parsers import PARSERS, detect_source_type

    store = get_kb_store()
    stype = req.source_type or detect_source_type(req.source)
    parser = PARSERS.get(stype)
    if not parser:
        return {"error": f"Unsupported source type: {stype}"}, 400

    try:
        text, metadata = parser(req.source)
    except Exception as e:
        return {"error": f"Failed to parse: {e}"}, 400

    if not text or not text.strip():
        return {"error": "No text content extracted"}, 400

    doc = await store.ingest(
        title=req.title,
        content=text,
        source=stype,
        origin=req.source if stype != "text" else "",
        metadata=metadata,
    )
    return {
        "status": "ingested",
        "document_id": doc.id,
        "title": doc.title,
        "source": doc.source,
        "chunk_count": doc.chunk_count,
    }


@app.post("/api/kb/ingest/upload")
async def kb_ingest_upload(file: UploadFile, title: str = Query("")):
    """Upload a file (PDF, DOCX, EML, TXT) to ingest into the knowledge base."""
    from cadence.knowledge.parsers import PARSERS, detect_source_type

    store = get_kb_store()
    filename = file.filename or "upload"
    stype = detect_source_type(filename)
    doc_title = title or filename

    data = await file.read()

    parser = PARSERS.get(stype)
    if not parser:
        return {"error": f"Unsupported file type: {stype}"}, 400

    try:
        text, metadata = parser(data)
    except Exception as e:
        return {"error": f"Failed to parse: {e}"}, 400

    if not text or not text.strip():
        return {"error": "No text content extracted from file"}, 400

    doc = await store.ingest(
        title=doc_title,
        content=text,
        source=stype,
        origin=filename,
        metadata=metadata,
    )
    return {
        "status": "ingested",
        "document_id": doc.id,
        "title": doc.title,
        "source": doc.source,
        "chunk_count": doc.chunk_count,
    }


@app.post("/api/kb/search")
async def kb_search(req: KBSearchRequest):
    """Search the knowledge base."""
    store = get_kb_store()
    results = await store.search(
        query=req.query,
        max_results=req.max_results,
        source_filter=req.source_filter,
    )
    return {
        "results": [
            {
                "chunk_id": r.chunk.id,
                "content": r.chunk.content,
                "similarity": round(r.similarity, 4),
                "relevance": round(r.relevance, 4),
                "document_id": r.chunk.document_id,
                "document_title": r.document.title if r.document else "",
                "source": r.document.source if r.document else "",
                "origin": r.document.origin if r.document else "",
            }
            for r in results
        ],
        "total": len(results),
    }


@app.get("/api/kb/documents")
async def kb_list_documents():
    """List all documents in the knowledge base."""
    store = get_kb_store()
    docs = await store.list_documents()
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "origin": d.origin,
            "chunk_count": d.chunk_count,
            "ingested_at": d.ingested_at,
        }
        for d in docs
    ]


@app.delete("/api/kb/documents/{document_id}")
async def kb_delete_document(document_id: str):
    """Delete a document and its chunks from the knowledge base."""
    store = get_kb_store()
    ok = await store.delete_document(document_id)
    return {"status": "deleted" if ok else "not_found"}


@app.get("/api/kb/stats")
async def kb_stats():
    """Get knowledge base statistics."""
    store = get_kb_store()
    return await store.stats()


# --- Streaming chat endpoint (SSE) ---

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events streaming endpoint for chat.

    Streams thinking steps, tool calls, and status in real-time,
    finishing with a 'done' event containing the full response.
    The running task is registered in _running_tasks so it can be cancelled
    via POST /api/chat/cancel/{session_id}.
    """
    agent_app = get_app()
    store = get_chat_store()
    conv_config = agent_app.config.conversation
    session_id = req.session_id or str(uuid.uuid4())[:8]
    history = list(store.get_session_history(session_id))

    collector = StreamCollector()

    async def _run_and_collect():
        start = time.time()
        try:
            response = await agent_app.run(req.message, conversation_history=history, images=req.images)
            duration_ms = (time.time() - start) * 1000

            # Persist exchange (with same compression/cap logic as the non-streaming endpoint)
            new_history = list(history)
            new_history.append({"role": "user", "content": req.message})
            new_history.append({"role": "assistant", "content": response})

            turn_count = len(new_history) // 2
            if conv_config.compression_enabled and turn_count > conv_config.compression_threshold:
                new_history = await _compress_history(agent_app, session_id, new_history)
            max_entries = conv_config.max_history_turns * 2
            if len(new_history) > max_entries:
                new_history = new_history[-max_entries:]

            store.save_session_history(
                session_id, new_history,
                summary=store.get_session_summary(session_id),
            )
            await collector.emit_done(
                full_response=response,
                session_id=session_id,
                duration_ms=duration_ms,
            )
        except asyncio.CancelledError:
            await collector.emit_error("Request cancelled by user.")
        except Exception as e:
            await collector.emit_error(str(e))
        finally:
            _running_tasks.pop(session_id, None)

    # Run agent in background; register task for cancellation
    task = asyncio.create_task(_run_and_collect())
    _running_tasks[session_id] = task

    # Patch the trace logger to emit thinking/status events in real-time
    original_log = agent_app.trace.log

    def _streaming_log(step):
        original_log(step)
        try:
            loop = asyncio.get_running_loop()
            if step.step_type == "thought":
                loop.create_task(collector.emit_thinking(step.content, step.agent_id))
            elif step.step_type == "action":
                loop.create_task(collector.emit_status(step.content, step.agent_id))
        except RuntimeError:
            pass

    agent_app.trace.log = _streaming_log

    async def _event_generator():
        try:
            async for event in collector:
                yield event.to_sse()
        finally:
            agent_app.trace.log = original_log

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/cancel/{session_id}")
async def cancel_chat(session_id: str):
    """Cancel a running chat request by session ID."""
    task = _running_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        _running_tasks.pop(session_id, None)
        return {"status": "cancelled", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# --- Checkpoint (human-in-the-loop) endpoints ---

class CheckpointResolveRequest(BaseModel):
    approved: bool
    response: str | None = None


@app.get("/api/checkpoints")
async def list_checkpoints(status: str | None = None):
    """List checkpoints. Use status='pending' to get only unresolved ones."""
    agent_app = get_app()
    mgr = agent_app.checkpoint_manager
    if not mgr:
        return {"error": "Checkpoints are disabled"}, 400

    if status == "pending":
        checkpoints = mgr.get_pending()
    else:
        checkpoints = mgr.get_all()

    return [cp.model_dump() for cp in checkpoints]


@app.post("/api/checkpoints/{checkpoint_id}/resolve")
async def resolve_checkpoint(checkpoint_id: str, req: CheckpointResolveRequest):
    """Approve or reject a pending checkpoint."""
    agent_app = get_app()
    mgr = agent_app.checkpoint_manager
    if not mgr:
        return {"error": "Checkpoints are disabled"}, 400

    cp = mgr.resolve(checkpoint_id, approved=req.approved, response=req.response)
    if not cp:
        return {"error": "Checkpoint not found or already resolved"}, 404
    return cp.model_dump()


# --- Message bus endpoints ---

@app.get("/api/bus/topics")
async def bus_topics():
    """List all message bus topics."""
    agent_app = get_app()
    bus = agent_app.message_bus
    if not bus:
        return {"error": "Message bus is disabled"}, 400
    return {"topics": bus.topics(), "stats": bus.stats()}


@app.get("/api/bus/messages/{topic}")
async def bus_messages(topic: str, limit: int = 20):
    """Read recent messages on a topic."""
    agent_app = get_app()
    bus = agent_app.message_bus
    if not bus:
        return {"error": "Message bus is disabled"}, 400
    messages = bus.peek(topic, limit=limit)
    return [m.model_dump() for m in messages]


# --- Knowledge graph endpoints ---

class GraphEntityRequest(BaseModel):
    name: str
    entity_type: str
    properties: dict | None = None


class GraphRelationRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: dict | None = None


@app.post("/api/graph/entities")
async def graph_add_entity(req: GraphEntityRequest):
    """Add an entity to the knowledge graph."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    entity = graph.add_entity(
        name=req.name,
        entity_type=req.entity_type,
        properties=req.properties,
    )
    return entity.model_dump()


@app.get("/api/graph/entities")
async def graph_find_entities(
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 20,
):
    """Search for entities in the knowledge graph."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    entities = graph.find_entities(name=name, entity_type=entity_type, limit=limit)
    return [e.model_dump() for e in entities]


@app.post("/api/graph/relationships")
async def graph_add_relation(req: GraphRelationRequest):
    """Add a relationship between entities."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    rel = graph.add_relationship(
        source_id=req.source_id,
        target_id=req.target_id,
        relation_type=req.relation_type,
        weight=req.weight,
        properties=req.properties,
    )
    if not rel:
        return {"error": "One or both entity IDs not found"}, 404
    return rel.model_dump()


@app.get("/api/graph/neighbors/{entity_id}")
async def graph_neighbors(entity_id: str, relation_type: str | None = None):
    """Get neighbors of an entity."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    result = graph.get_neighbors(entity_id, relation_type=relation_type)
    return {
        "entities": [e.model_dump() for e in result.entities],
        "relationships": [r.model_dump() for r in result.relationships],
    }


@app.get("/api/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    return graph.stats()


@app.delete("/api/graph/entities/{entity_id}")
async def graph_delete_entity(entity_id: str):
    """Delete an entity and its relationships."""
    agent_app = get_app()
    graph = agent_app.knowledge_graph
    if not graph:
        return {"error": "Knowledge graph is disabled"}, 400
    ok = graph.delete_entity(entity_id)
    return {"status": "deleted" if ok else "not_found"}


# --- Learning endpoints ---

@app.get("/api/learning/stats")
async def learning_stats():
    """Get cross-session learning statistics."""
    agent_app = get_app()
    store = agent_app.learning_store
    if not store:
        return {"error": "Learning is disabled"}, 400
    return store.get_stats()


@app.get("/api/learning/insights/{task_type}")
async def learning_insights(task_type: str, limit: int = 5):
    """Get learning insights for a task type."""
    agent_app = get_app()
    store = agent_app.learning_store
    if not store:
        return {"error": "Learning is disabled"}, 400
    insights = store.get_insights(task_type, limit=limit)
    return [i.model_dump() for i in insights]


# --- WebSocket for live trace streaming ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cid = id(ws)
    client = WSClient(ws)
    _ws_clients[cid] = client
    # Start a drain task for this client's message queue
    _ws_drain_tasks[cid] = asyncio.create_task(client.drain())
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _remove_ws_client(cid)


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
