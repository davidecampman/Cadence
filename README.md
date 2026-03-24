# Sentinel

A model-agnostic multi-agent framework with structured planning, tiered memory, and parallel task execution.

Sentinel enables autonomous agents to break down complex tasks into dependency graphs, delegate work to specialist agents, and coordinate results — all while maintaining persistent memory and reasoning traces.

## Features

- **Multi-Agent Orchestration** — Task DAG with dependency resolution, specialist agent roles (researcher, coder, reviewer), parallel execution, and loop detection
- **Smart Model Routing** — Two-tier model strategy (fast/strong), automatic task classification, fallback chains, and per-model success tracking
- **Tiered Memory** — ChromaDB-backed vector store with time-decay relevance scoring, importance weighting, and namespace isolation
- **Skill System** — Declarative SKILL.md format with versioning, dependency resolution, and auto-discovery
- **Built-in Tools** — File operations, code execution (sandboxed), web fetching, memory management, and agent delegation
- **Security & Sandboxing** — Permission tiers, execution timeouts, resource limits, and blocked command lists
- **Reasoning Traces** — Step-by-step JSONL logging with WebSocket streaming to the frontend
- **Web UI** — React frontend with chat, tool/skill browsers, config panel, and live reasoning trace

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core | Python 3.11+, LiteLLM, Pydantic 2.0+ |
| API | FastAPI, Uvicorn, WebSocket |
| Memory | ChromaDB |
| Frontend | React 19, TypeScript, Vite |
| Testing | pytest, pytest-asyncio, Ruff |
| Deployment | Docker, Docker Compose |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+ (for frontend)

### Installation

```bash
# Install Python package in dev mode
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend && npm ci && cd ..

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Run the CLI

```bash
sentinel
```

Commands: `/skills`, `/trace`, `/config`, `/quit`

### Run the API Server

```bash
sentinel-server
```

This serves the REST API at `http://localhost:8000/api`, WebSocket at `ws://localhost:8000/ws`, and the React frontend at `http://localhost:8000`.

### Run with Docker

```bash
docker-compose up --build
```

## Project Structure

```
sentinel/
├── agents/          # Multi-agent orchestration and task DAG execution
├── core/            # Agent loop, config, LLM abstraction, keystore, types
├── memory/          # ChromaDB-backed tiered memory with time decay
├── routing/         # Smart model routing with fallback chains
├── skills/          # SKILL.md parser with dependency resolution
├── tools/           # Built-in tools (file ops, code exec, web, memory, delegation)
├── api.py           # FastAPI REST + WebSocket endpoints
├── cli.py           # Interactive REPL
└── server.py        # API server entry point
config/
└── default.yaml     # Default configuration
frontend/            # React + TypeScript web UI
skills/              # Example skill definitions
tests/               # Unit tests
```

## Configuration

Configuration lives in `config/default.yaml` and can be overridden with environment variables:

```bash
SENTINEL_MODELS_STRONG=gpt-4o
SENTINEL_AGENTS_MAX_DEPTH=3
SENTINEL_MEMORY_DECAY_RATE=0.1
```

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.strong` | claude-sonnet-4-20250514 | Model for complex reasoning and code |
| `models.fast` | claude-haiku-4-5-20251001 | Model for planning and simple tasks |
| `agents.max_parallel` | 4 | Max concurrent agents |
| `agents.max_iterations_per_task` | 25 | Circuit breaker per task |
| `memory.decay_rate` | 0.05 | Relevance decay per day |
| `execution.timeout_seconds` | 120 | Per-execution timeout |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message |
| GET | `/config` | Get current configuration |
| POST | `/config` | Update configuration |
| GET | `/tools` | List available tools |
| GET | `/skills` | List loaded skills |
| GET | `/health` | Health check |
| WS | `/ws` | Live reasoning trace stream |
| GET | `/keys` | List stored API key providers |
| POST | `/keys` | Store an API key |
| DELETE | `/keys/{provider}` | Remove a stored key |

## Testing

```bash
pytest tests/ -v
```

## License

See [LICENSE](LICENSE) for details.
