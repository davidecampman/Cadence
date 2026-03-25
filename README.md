# Sentinel

A model-agnostic multi-agent framework with structured planning, tiered memory, and parallel task execution.

Sentinel enables autonomous agents to break down complex tasks into dependency graphs, delegate work to specialist agents, and coordinate results — all while maintaining persistent memory and reasoning traces.

## Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph Clients
        CLI[CLI REPL]
        WebUI[React Web UI]
    end

    subgraph API Layer
        FastAPI[FastAPI Server]
        WS[WebSocket]
    end

    subgraph Orchestration
        Orch[Orchestrator]
        DAG[Task DAG]
    end

    subgraph Agents
        A1[Researcher Agent]
        A2[Coder Agent]
        A3[Reviewer Agent]
        A4[General Agent]
    end

    subgraph Core Services
        LLM[LLM Interface<br/>LiteLLM]
        Router[Smart Router]
        Memory[Memory Store<br/>ChromaDB]
        Skills[Skill Loader]
        Tools[Tool Registry]
        Trace[Trace Logger]
    end

    subgraph LLM Providers
        Claude[Claude]
        GPT[GPT-4]
        Bedrock[AWS Bedrock]
    end

    CLI --> Orch
    WebUI --> FastAPI
    FastAPI --> Orch
    WS -.->|trace stream| WebUI

    Orch --> DAG
    DAG --> A1 & A2 & A3 & A4

    A1 & A2 & A3 & A4 --> Tools
    A1 & A2 & A3 & A4 --> LLM
    A1 & A2 & A3 & A4 --> Memory

    LLM --> Router
    Router --> Claude & GPT & Bedrock

    Orch --> Skills
    Orch --> Trace
    Trace -.-> WS
```

### Request Lifecycle

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI / CLI
    participant Orch as Orchestrator
    participant LLM as LLM (Fast)
    participant DAG as Task DAG
    participant Agent as Agent(s)
    participant LLM2 as LLM (Strong)
    participant Tools as Tools
    participant WS as WebSocket

    User->>API: Send message
    API->>Orch: run(message)

    rect rgb(235, 245, 255)
        Note over Orch,LLM: Phase 1 — Planning
        Orch->>LLM: Decompose request
        LLM-->>Orch: Task list + dependencies
        Orch->>DAG: Build task graph
    end

    rect rgb(235, 255, 240)
        Note over Orch,Tools: Phase 2 — Execution
        loop For each ready task (parallel)
            DAG->>Agent: Spawn specialist agent
            loop Think → Act → Observe
                Agent->>LLM2: Messages + tool defs
                LLM2-->>Agent: Response + tool calls
                Agent->>Tools: Execute tool calls
                Tools-->>Agent: Tool results
                Agent-->>WS: Trace steps
            end
            Agent-->>DAG: Task result
        end
    end

    rect rgb(255, 245, 235)
        Note over Orch,LLM2: Phase 3 — Synthesis
        Orch->>LLM2: Combine all task results
        LLM2-->>Orch: Final response
    end

    rect rgb(250, 240, 255)
        Note over Orch,LLM2: Phase 4 — Evaluation
        Orch->>LLM2: Self-check quality
        LLM2-->>Orch: Pass / revision needed
    end

    Orch-->>API: Response + trace
    API-->>User: Final answer
```

### Agent Think → Act → Observe Loop

```mermaid
flowchart TD
    Start([Agent receives task]) --> BuildCtx[Build context<br/>system prompt + history + task]
    BuildCtx --> CheckBudget{Budget &<br/>loop check}
    CheckBudget -->|Over limit| Bail([Return partial result])
    CheckBudget -->|OK| CallLLM[Call LLM with<br/>messages + tool definitions]
    CallLLM --> HasTools{Tool calls<br/>in response?}
    HasTools -->|No| Done([Return final answer])
    HasTools -->|Yes| ExecTools[Execute each tool call]
    ExecTools --> Record[Record results in history]
    Record --> LogTrace[Log trace step via WebSocket]
    LogTrace --> CheckBudget
```

### Smart Model Routing

```mermaid
flowchart LR
    Task([Incoming Task]) --> Classify{Classify task type}
    Classify -->|plan / evaluate| Fast[Fast Model<br/>Claude Haiku]
    Classify -->|code / reason| Strong[Strong Model<br/>Claude Sonnet]
    Fast --> Track[Track success rate<br/>& latency]
    Strong --> Track
    Track --> Fallback{Model failed?}
    Fallback -->|Yes| Next[Try next in<br/>fallback chain]
    Fallback -->|No| Result([Return response])
    Next --> Track
```

### Memory System

```mermaid
flowchart TD
    subgraph Write Path
        AgentW([Agent]) -->|save| MemTool[Memory Save Tool]
        MemTool --> Embed[Generate embedding]
        Embed --> Store[(ChromaDB)]
    end

    subgraph Read Path
        AgentR([Agent]) -->|query| QueryTool[Memory Query Tool]
        QueryTool --> VecSearch[Vector similarity search]
        VecSearch --> Store
        Store --> Score[Apply scoring:<br/>similarity × importance × e^−decay·age]
        Score --> TopK[Return top K results]
        TopK --> AgentR
    end

    subgraph Namespaces
        NS1[shared — all agents]
        NS2[project:X — project-scoped]
        NS3[agent:X — private to agent]
    end

    Store --- NS1 & NS2 & NS3
```

### System Component Map

```mermaid
graph LR
    subgraph sentinel/core
        agent[agent.py<br/>Think→Act loop]
        config[config.py<br/>YAML + env vars]
        llm[llm.py<br/>LiteLLM wrapper]
        types[types.py<br/>Pydantic models]
        trace[trace.py<br/>JSONL logger]
        keystore[keystore.py<br/>Encrypted keys]
    end

    subgraph sentinel/agents
        orch[orchestrator.py<br/>DAG execution]
    end

    subgraph sentinel/tools
        base[base.py — Registry]
        file_ops[file_ops.py]
        code_exec[code_execution.py]
        web[web.py]
        browser_tool[browser.py<br/>Playwright]
        mem_tools[memory_tools.py]
        kb_tools[knowledge_tools.py]
        prompt_tools[prompt_tools.py]
        delegate[delegate.py]
        git[git_ops.py]
        other[+ 6 more tools]
    end

    subgraph sentinel/memory
        memstore[store.py<br/>ChromaDB + decay]
    end

    subgraph sentinel/knowledge
        kbstore[store.py<br/>Document search]
        parsers[parsers.py<br/>PDF/DOCX/email]
    end

    subgraph sentinel/prompts
        prompt_store[store.py<br/>Modification DB]
        evolution[evolution.py<br/>LLM reflection]
    end

    subgraph sentinel/routing
        router[router.py<br/>Model selection]
    end

    subgraph sentinel/skills
        loader[loader.py<br/>SKILL.md parser]
    end

    subgraph sentinel/storage
        chat_store[chat_store.py<br/>SQLite persistence]
    end

    subgraph API
        api[api.py — FastAPI + WS]
        app[app.py — Bootstrap]
        cli[cli.py — REPL]
    end

    subgraph frontend
        react[React 19 + TS + Vite]
    end

    app --> orch & agent & config & llm & memstore & router & loader & base
    orch --> agent
    agent --> llm & base & memstore
    llm --> router
    kb_tools --> kbstore
    kbstore --> parsers
    prompt_tools --> prompt_store & evolution
    api --> app & chat_store
    cli --> app
    react -->|HTTP + WS| api
```

## Features

- **Multi-Agent Orchestration** — Task DAG with dependency resolution, specialist agent roles (researcher, coder, reviewer), parallel execution, and loop detection
- **Smart Model Routing** — Two-tier model strategy (fast/strong), automatic task classification, fallback chains, and per-model success tracking
- **Tiered Memory** — ChromaDB-backed vector store with time-decay relevance scoring, importance weighting, and namespace isolation
- **Knowledge Base Ingestion** — Ingest and search across PDFs, DOCX, emails (.eml), web pages, and plain text with automatic chunking and semantic search
- **Self-Modifying Prompts** — LLM-driven prompt evolution with reflection after tasks, performance-based modifications, version history, and rollback
- **Skill System** — Declarative SKILL.md format with versioning, dependency resolution, and auto-discovery
- **Built-in Tools** — File operations, code execution (sandboxed), web fetching, memory management, browser automation, knowledge base, prompt management, and agent delegation
- **Browser Automation** — Playwright-powered headless browsing with navigation, clicking, form filling, screenshots, and structured data extraction
- **Security & Sandboxing** — Permission tiers (read-only, standard, privileged, unrestricted), execution timeouts, resource limits, path allowlists, and blocked command lists
- **Persistent Chat Storage** — SQLite-backed chat and session history that survives server restarts, with automatic localStorage migration and offline fallback
- **Conversation Context Management** — Configurable history window with automatic LLM-based compression of older turns
- **Reasoning Traces** — Step-by-step JSONL logging with WebSocket streaming to the frontend
- **Web UI** — React frontend with multi-chat sidebar, tabbed config panel (token budget, agents, memory), tool/skill browsers, and live reasoning trace

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core | Python 3.11+, LiteLLM, Pydantic 2.0+ |
| API | FastAPI, Uvicorn, WebSocket |
| Memory | ChromaDB |
| Knowledge | ChromaDB, PyPDF, python-docx, BeautifulSoup4 |
| Storage | SQLite (WAL mode) |
| Browser | Playwright (optional) |
| Cloud | AWS Bedrock via boto3 (optional) |
| Frontend | React 19, TypeScript, Vite |
| Testing | pytest, pytest-asyncio, Ruff |
| Deployment | pip, npm |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+ (for frontend)

### Installation

```bash
# Install Python package in dev mode
pip install -e ".[dev]"

# Install with browser automation support (optional)
pip install -e ".[dev,browser]"
playwright install chromium

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

## Project Structure

```
sentinel/
├── agents/          # Multi-agent orchestration and task DAG execution
├── core/            # Agent loop, config, LLM abstraction, keystore, types
├── knowledge/       # Knowledge base ingestion, document parsing, and semantic search
├── memory/          # ChromaDB-backed tiered memory with time decay
├── prompts/         # Self-modifying prompt evolution and version tracking
├── routing/         # Smart model routing with fallback chains
├── skills/          # SKILL.md parser with dependency resolution
├── storage/         # SQLite-backed persistent chat and session storage
├── tools/           # Built-in tools (see Tools section below)
├── api.py           # FastAPI REST + WebSocket endpoints
├── cli.py           # Interactive REPL
└── server.py        # API server entry point
config/
└── default.yaml     # Default configuration
data/                # Runtime data (SQLite DB, traces, memory vectors)
frontend/            # React + TypeScript web UI
skills/              # Example skill definitions
tests/               # Unit and integration tests
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
| `models.strong` | claude-sonnet-4-5-20250514 | Model for complex reasoning and code |
| `models.fast` | claude-haiku-4-5-20251001 | Model for planning and simple tasks |
| `models.embedding` | text-embedding-3-small | Model for memory and KB embeddings |
| `agents.max_parallel` | 4 | Max concurrent agents |
| `agents.max_iterations_per_task` | 25 | Circuit breaker per task |
| `budget.max_tokens_per_task` | 100000 | Per-task token ceiling |
| `budget.max_tokens_per_session` | 500000 | Per-session token ceiling |
| `memory.decay_rate` | 0.05 | Relevance decay per day |
| `conversation.max_history_turns` | 50 | Max user+assistant pairs to retain |
| `conversation.compression_enabled` | true | Summarize older context automatically |
| `prompt_evolution.enabled` | true | Enable self-modifying prompt system |
| `execution.timeout_seconds` | 120 | Per-execution timeout |

## Built-in Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| ReadFile, WriteFile, ListFiles, SearchFiles | STANDARD | File operations with path safety checks |
| CodeExecution | PRIVILEGED | Sandboxed code/shell execution with resource limits |
| WebFetch | STANDARD | HTTP requests and web page fetching |
| BrowseWeb, BrowserClick, BrowserForm, BrowserScreenshot, BrowserExtract | PRIVILEGED | Playwright-powered headless browser automation |
| MemorySave, MemoryQuery, MemoryDelete | STANDARD | Tiered memory with namespace isolation |
| KBIngest, KBSearch, KBList, KBDelete | STANDARD | Knowledge base document ingestion and search |
| PromptView, PromptModify, PromptHistory, PromptRollback | STANDARD | Self-modifying prompt management |
| Delegate | STANDARD | Spawn sub-agents for parallel task execution |
| GitOps | PRIVILEGED | Git operations (status, diff, commit, branch) |
| Database | PRIVILEGED | Database queries and schema inspection |
| HTTPClient | STANDARD | Structured HTTP API calls |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message |
| GET | `/chats` | List all persisted chats |
| GET | `/chats/{id}` | Get a chat with all messages |
| POST | `/chats` | Create a new chat |
| PUT | `/chats/{id}` | Update chat metadata |
| DELETE | `/chats/{id}` | Delete a chat and its messages |
| POST | `/chats/{id}/messages` | Add a message to a chat |
| GET | `/config` | Get current configuration |
| PUT | `/config` | Update configuration |
| GET | `/tools` | List available tools |
| GET | `/skills` | List loaded skills |
| GET | `/health` | Health check |
| WS | `/ws` | Live reasoning trace stream |
| GET | `/keys` | List stored API key providers |
| POST | `/keys` | Store an API key |
| DELETE | `/keys/{provider}` | Remove a stored key |
| POST | `/api/kb/ingest` | Ingest a document from path, URL, or text |
| POST | `/api/kb/ingest/upload` | Ingest an uploaded file |
| POST | `/api/kb/search` | Semantic search across knowledge base |
| GET | `/api/kb/documents` | List all ingested documents |
| DELETE | `/api/kb/documents/{id}` | Delete a document |
| GET | `/api/kb/stats` | Knowledge base statistics |
| GET | `/api/files/download` | Download a project file |
| GET | `/api/files/reveal` | Open file location in system file manager |

## Testing

```bash
pytest tests/ -v
```

## License

See [LICENSE](LICENSE) for details.
