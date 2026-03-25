# Cadence

A model-agnostic multi-agent framework with structured planning, tiered memory, and parallel task execution.

Cadence enables autonomous agents to break down complex tasks into dependency graphs, delegate work to specialist agents, and coordinate results — all while maintaining persistent memory and reasoning traces.

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
        SSE[SSE Streaming]
    end

    subgraph Orchestration
        Orch[Orchestrator]
        DAG[Task DAG]
        Checkpoint[Checkpoint Manager]
    end

    subgraph Agents
        A1[Researcher Agent]
        A2[Coder Agent]
        A3[Reviewer Agent]
        A4[General Agent]
    end

    subgraph Communication
        Bus[Message Bus<br/>Pub/Sub]
        Stream[Stream Collector]
    end

    subgraph Core Services
        LLM[LLM Interface<br/>LiteLLM]
        Router[Smart Router]
        Memory[Memory Store<br/>ChromaDB]
        Skills[Skill Loader]
        Tools[Tool Registry]
        Trace[Trace Logger]
        MultiModal[Multi-Modal<br/>Vision Support]
    end

    subgraph Knowledge & Learning
        KB[Knowledge Base<br/>ChromaDB]
        KG[Knowledge Graph<br/>NetworkX]
        Learning[Learning Store<br/>SQLite]
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
    SSE -.->|token stream| WebUI

    Orch --> DAG
    Orch --> Checkpoint
    Checkpoint -.->|approval request| FastAPI
    DAG --> A1 & A2 & A3 & A4

    A1 & A2 & A3 & A4 --> Tools
    A1 & A2 & A3 & A4 --> LLM
    A1 & A2 & A3 & A4 --> Memory
    A1 & A2 & A3 & A4 <--> Bus
    A1 & A2 & A3 & A4 --> Stream

    LLM --> Router
    LLM --> MultiModal
    Router --> Claude & GPT & Bedrock

    Tools --> KB & KG
    Orch --> Skills & Trace & Learning
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

### Agent Message Bus

```mermaid
flowchart TD
    subgraph Agents
        A1([Agent A])
        A2([Agent B])
        A3([Agent C])
    end

    subgraph Message Bus
        Pub[Publish]
        Topics[Topic Router]
        Sub[Subscribers]
        History[(Message History)]
    end

    subgraph Topics
        T1[discovery]
        T2[coordination]
        T3[status]
        T4[error]
        T5[custom topics...]
    end

    A1 -->|publish| Pub
    Pub --> Topics
    Topics --> T1 & T2 & T3 & T4 & T5
    T1 & T2 & T3 & T4 & T5 --> Sub
    Sub -->|notify| A2 & A3
    Pub --> History

    subgraph Request-Reply
        Req[Request with timeout]
        Rep[Reply on reply_to topic]
    end

    A1 -->|request| Req
    Req --> Topics
    A2 -->|reply| Rep
    Rep --> A1
```

### Streaming & Checkpoints

```mermaid
flowchart LR
    subgraph Agent Execution
        Agent([Agent]) --> Collector[Stream Collector]
    end

    subgraph Event Types
        Collector -->|token| T[Text tokens]
        Collector -->|thinking| Th[Reasoning steps]
        Collector -->|tool_start| TS[Tool invocations]
        Collector -->|tool_result| TR[Tool results]
        Collector -->|done| D[Final response]
    end

    subgraph SSE Delivery
        T & Th & TS & TR & D --> SSE[Server-Sent Events]
        SSE --> Client([Web UI / Client])
    end

    subgraph Checkpoint Flow
        Agent2([Agent]) -->|needs approval| CP[Checkpoint Manager]
        CP -->|pending| API[REST API]
        API -->|resolve| CP
        CP -->|approved/rejected| Agent2
    end
```

### Cross-Session Learning

```mermaid
flowchart TD
    subgraph Recording
        Agent([Agent completes task]) --> Record[Record strategy]
        Record --> DB[(SQLite<br/>Learning Store)]
    end

    subgraph Strategy Record
        DB --> Fields[task_type + strategy<br/>tools_used + model<br/>outcome + iterations<br/>tokens + duration]
    end

    subgraph Insights
        Query([New task arrives]) --> Classify[Classify task type]
        Classify --> Lookup[Query past strategies]
        Lookup --> DB
        DB --> Analyze[Rank by success rate]
        Analyze --> Insight[LearningInsight<br/>confidence + recommendation]
        Insight --> Agent2([Agent])
    end

    subgraph Analytics
        DB --> Stats[Stats by task type]
        DB --> BestTools[Best tools per task]
    end
```

### Knowledge Graph

```mermaid
flowchart TD
    subgraph Entities
        E1[function:parse_config]
        E2[class:ConfigManager]
        E3[module:cadence.core]
        E4[file:config.py]
        E5[concept:configuration]
    end

    subgraph Relationships
        E1 -->|belongs_to| E2
        E2 -->|contains| E1
        E2 -->|defined_in| E4
        E4 -->|part_of| E3
        E2 -->|implements| E5
    end

    subgraph Operations
        Add[Add Entity/Relation]
        Search[Find Entities]
        Neighbors[Get Neighbors]
        Path[Find Path]
        Subgraph[Extract Subgraph]
    end

    subgraph Storage
        Graph[(NetworkX DiGraph)] --> JSON[JSON persistence]
    end

    Add & Search & Neighbors & Path & Subgraph --> Graph
```

### System Component Map

```mermaid
graph LR
    subgraph cadence/core
        agent[agent.py<br/>Think→Act loop]
        config[config.py<br/>YAML + env vars]
        llm[llm.py<br/>LiteLLM wrapper]
        types[types.py<br/>Pydantic models]
        trace[trace.py<br/>JSONL logger]
        keystore[keystore.py<br/>Encrypted keys]
        message_bus[message_bus.py<br/>Pub/Sub]
        streaming[streaming.py<br/>SSE events]
        checkpoint[checkpoint.py<br/>Human-in-the-loop]
        multimodal[multimodal.py<br/>Vision support]
    end

    subgraph cadence/agents
        orch[orchestrator.py<br/>DAG execution]
    end

    subgraph cadence/tools
        base[base.py — Registry]
        file_ops[file_ops.py]
        code_exec[code_execution.py]
        web[web.py]
        browser_tool[browser.py<br/>Playwright]
        mem_tools[memory_tools.py]
        kb_tools[knowledge_tools.py]
        graph_tools[graph_tools.py]
        prompt_tools[prompt_tools.py]
        bus_tools[message_bus_tools.py]
        cp_tools[checkpoint_tools.py]
        vision_tools[vision.py]
        delegate[delegate.py]
        git[git_ops.py]
        other[+ 5 more tools]
    end

    subgraph cadence/memory
        memstore[store.py<br/>ChromaDB + decay]
    end

    subgraph cadence/knowledge
        kbstore[store.py<br/>Document search]
        parsers[parsers.py<br/>PDF/DOCX/email]
        kg[graph.py<br/>Knowledge graph]
    end

    subgraph cadence/learning
        learn_store[store.py<br/>Strategy tracking]
    end

    subgraph cadence/prompts
        prompt_store[store.py<br/>Modification DB]
        evolution[evolution.py<br/>LLM reflection]
    end

    subgraph cadence/routing
        router[router.py<br/>Model selection]
    end

    subgraph cadence/skills
        loader[loader.py<br/>SKILL.md parser]
    end

    subgraph cadence/storage
        chat_store[chat_store.py<br/>SQLite persistence]
    end

    subgraph API
        api[api.py — FastAPI + WS + SSE]
        app[app.py — Bootstrap]
        cli[cli.py — REPL]
    end

    subgraph frontend
        react[React 19 + TS + Vite]
    end

    app --> orch & agent & config & llm & memstore & router & loader & base & message_bus & checkpoint & learn_store
    orch --> agent
    agent --> llm & base & memstore & message_bus & streaming
    llm --> router & multimodal
    kb_tools --> kbstore
    graph_tools --> kg
    kbstore --> parsers
    prompt_tools --> prompt_store & evolution
    bus_tools --> message_bus
    cp_tools --> checkpoint
    api --> app & chat_store & streaming
    cli --> app
    react -->|HTTP + WS + SSE| api
```

## Features

- **Multi-Agent Orchestration** — Task DAG with dependency resolution, specialist agent roles (researcher, coder, reviewer), parallel execution, and loop detection
- **Smart Model Routing** — Two-tier model strategy (fast/strong), automatic task classification, fallback chains, and per-model success tracking
- **Agent Message Bus** — Async pub/sub inter-agent communication with topic-based routing, priority levels (low/normal/high/urgent), request-reply patterns with timeouts, and message history
- **Streaming Responses** — Server-Sent Events (SSE) streaming with real-time token delivery, tool execution events, reasoning steps, and status updates via async event queues
- **Human-in-the-Loop Checkpoints** — Approval workflows that pause agent execution for human review, with checkpoint types (approval, clarification, confirmation), configurable timeouts, and REST API resolution
- **Tiered Memory** — ChromaDB-backed vector store with time-decay relevance scoring, importance weighting, and namespace isolation
- **Knowledge Base Ingestion** — Ingest and search across PDFs, DOCX, emails (.eml), web pages, and plain text with automatic chunking and semantic search
- **Knowledge Graph** — NetworkX-backed directed graph for structured entity-relationship modeling, with path finding, subgraph extraction, neighbor traversal, and JSON persistence
- **Cross-Session Learning** — SQLite-backed strategy tracking across sessions with task classification, success rate analytics, tool effectiveness ranking, and confidence-scored recommendations
- **Multi-Modal Input** — Vision support for images (PNG, JPEG, GIF, WebP) from files, base64, or URLs, with automatic vision model detection across Claude, GPT-4, and Gemini
- **Self-Modifying Prompts** — LLM-driven prompt evolution with reflection after tasks, performance-based modifications, version history, and rollback
- **Skill System** — Declarative SKILL.md format with versioning, dependency resolution, and auto-discovery
- **Built-in Tools** — 30+ tools for file operations, code execution (sandboxed), web fetching, memory, browser automation, knowledge base/graph, prompt management, learning insights, message bus, and agent delegation
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
| API | FastAPI, Uvicorn, WebSocket, Server-Sent Events |
| Memory | ChromaDB |
| Knowledge | ChromaDB, NetworkX, PyPDF, python-docx, BeautifulSoup4 |
| Learning | SQLite (WAL mode) |
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
cadence
```

Commands: `/skills`, `/trace`, `/config`, `/quit`

### Run the API Server

```bash
cadence-server
```

This serves the REST API at `http://localhost:8000/api`, WebSocket at `ws://localhost:8000/ws`, and the React frontend at `http://localhost:8000`.

## Project Structure

```
cadence/
├── agents/          # Multi-agent orchestration and task DAG execution
├── core/            # Agent loop, config, LLM, keystore, message bus, streaming, checkpoints, multi-modal
├── knowledge/       # Knowledge base ingestion, document parsing, knowledge graph, and semantic search
├── learning/        # Cross-session strategy tracking and analytics
├── memory/          # ChromaDB-backed tiered memory with time decay
├── prompts/         # Self-modifying prompt evolution and version tracking
├── routing/         # Smart model routing with fallback chains
├── skills/          # SKILL.md parser with dependency resolution
├── storage/         # SQLite-backed persistent chat and session storage
├── tools/           # 30+ built-in tools (see Tools section below)
├── api.py           # FastAPI REST + WebSocket + SSE endpoints
├── cli.py           # Interactive REPL
└── server.py        # API server entry point
config/
└── default.yaml     # Default configuration
data/                # Runtime data (SQLite DB, traces, memory vectors, knowledge graph)
frontend/            # React + TypeScript web UI
skills/              # Example skill definitions
tests/               # Unit and integration tests
```

## Configuration

Configuration lives in `config/default.yaml` and can be overridden with environment variables:

```bash
CADENCE_MODELS_STRONG=gpt-4o
CADENCE_AGENTS_MAX_DEPTH=3
CADENCE_MEMORY_DECAY_RATE=0.1
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
| GraphAddEntity, GraphAddRelation, GraphQuery | STANDARD | Knowledge graph entity and relationship management |
| PromptView, PromptModify, PromptHistory, PromptRollback | STANDARD | Self-modifying prompt management |
| BusPublish, BusPeek | STANDARD | Inter-agent message bus communication |
| RequestApproval | STANDARD | Human-in-the-loop checkpoint requests |
| LearningInsights, LearningStats | READ_ONLY | Cross-session strategy analytics and recommendations |
| Screenshot, ImageDescribe | STANDARD | Screen capture and image analysis for multi-modal input |
| Delegate | STANDARD | Spawn sub-agents for parallel task execution |
| GitOps | PRIVILEGED | Git operations (status, diff, commit, branch) |
| Database | PRIVILEGED | Database queries and schema inspection |
| HTTPClient | STANDARD | Structured HTTP API calls |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message |
| POST | `/api/chat/stream` | Send a message with SSE streaming response |
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
| POST | `/api/graph/entities` | Create a knowledge graph entity |
| GET | `/api/graph/entities` | Search entities by name or type |
| POST | `/api/graph/relationships` | Create a relationship between entities |
| GET | `/api/graph/neighbors/{id}` | Get neighboring entities |
| GET | `/api/graph/stats` | Knowledge graph statistics |
| DELETE | `/api/graph/entities/{id}` | Delete an entity and its edges |
| GET | `/api/bus/topics` | List message bus topics and stats |
| GET | `/api/bus/messages/{topic}` | Read recent messages on a topic |
| GET | `/api/checkpoints` | List checkpoints (filter by status) |
| POST | `/api/checkpoints/{id}/resolve` | Approve or reject a checkpoint |
| GET | `/api/learning/stats` | Aggregate learning statistics |
| GET | `/api/learning/insights/{type}` | Get strategy recommendations for task type |
| GET | `/api/files/download` | Download a project file |
| GET | `/api/files/reveal` | Open file location in system file manager |

## Testing

```bash
pytest tests/ -v
```

## License

See [LICENSE](LICENSE) for details.
