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
