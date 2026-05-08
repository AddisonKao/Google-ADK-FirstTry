# Google ADK LLMOps PoC

A production-grade proof-of-concept demonstrating an **event-driven AI agent** with RAG, full observability, prompt management, and a multi-level eval CI pipeline — all running locally with Docker.

The domain is insurance customer service, powered by a fictional Taiwanese insurer "太平盛世人壽".

---

## Architecture

```
Browser (Chat UI)
    │ POST /conversations/{id}/turns
    ▼
FastAPI (api/)          ← REST + SSE streaming + conversation history (postgres)
    │ Kafka: agent-input
    ▼
ADK Consumer (agent/)   ← Google ADK LlmAgent, DatabaseSessionService, ThreadPoolExecutor
    │ retrieve_tool → pgvector RAG (docs/)
    │ LLM: Gemini or OpenAI-compat (e.g. RDSec)
    │ Langfuse tracing callbacks
    │ Kafka: agent-output
    ▼
FastAPI                 ← SSE push to browser
```

### Observability

```
agent/ + api/ → OTel Collector → Tempo (traces) + Prometheus (metrics) → Grafana
                              → Langfuse (LLM-specific: prompt versions, token counts)
```

---

## Quick Start

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY (or OPENAI_API_BASE + OPENAI_API_KEY for OpenAI-compat)
# Optionally fill in LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY

./up.sh
```

**First time only** — populate the insurance knowledge base:
```bash
docker compose run --rm agent-consumer python ingest.py
```

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:8000 |
| Langfuse | http://localhost:3030 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## LLM Mode

The agent supports two interchangeable backends:

| Mode | Activate by setting | Default model |
|------|---------------------|---------------|
| Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| OpenAI-compatible | `OPENAI_API_BASE` + `OPENAI_API_KEY` | `gpt-4o` |

When both are set, OpenAI-compat takes priority. Same dual-mode applies to embeddings.

> **Note**: `gemini-2.5-flash` is recommended over `gemini-2.5-flash-lite`. The lite model's free tier rate limits cause intermittent silent failures on the second LLM call after a tool use. An application-level retry (3 attempts, 2s delay) mitigates this, but `gemini-2.5-flash` is more reliable.

---

## RAG Knowledge Base

Five Chinese Markdown files under `docs/` form the insurance knowledge base:

| File | Content |
|------|---------|
| `products.md` | 4 insurance products with premiums and coverage |
| `premium_rates.md` | Age/gender rate tables |
| `claims_process.md` | Claims procedures and document checklists |
| `coverage_rules.md` | Underwriting rules, exclusions, policy conditions |
| `faq.md` | 10+ Q&A covering billing, claims, policy management |

The `ingest.py` script splits each file by `##` headings into chunks (max 1000 chars), embeds with `gemini-embedding-001` (768-dim), and stores in `rag.documents` via pgvector.

The agent's `retrieve_tool` performs pure vector similarity search and returns the top-K chunks as a dict `{"result": "..."}`.

---

## Prompt Management

System instructions are managed in **Langfuse** under the key `kafka-agent-system`:

1. Go to http://localhost:3030 → **Prompts** → `kafka-agent-system`
2. Edit and add a new version (stays as draft)
3. Test in a local environment by setting `LANGFUSE_ENVIRONMENT=staging`
4. Promote to production by moving the `production` label

The agent fetches the prompt on every request (SDK caches 5 seconds). No restart needed after a prompt change.

---

## Testing

Install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

### Layer 1 — Unit Tests (fast, offline)
```bash
# Local
pytest tests/unit/ -v

# Docker (no local Python env needed)
docker compose run --rm agent-consumer pytest tests/unit/ -v
```

### Layer 2 — Trajectory Tests (requires LLM API key)
```bash
docker compose run --rm \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  agent-consumer pytest tests/evals/ -v
```

Verifies that:
- Insurance questions → `retrieve_tool` IS called
- General questions → `retrieve_tool` is NOT called

> **Note**: Uses `InMemoryRunner` + mock `retrieve_tool` instead of `AgentEvaluator` due to ADK bug [#5410](https://github.com/google/adk-python/issues/5410) (`tool_trajectory_avg_score` always 0.0). Revert to `AgentEvaluator` once PR #5417 is merged.

### Layer 3 — LLM-as-Judge Eval
```bash
docker compose run --rm agent-consumer python eval/run_eval.py
```

Runs 8 golden cases through the full agent, scores each with an LLM judge, checks regression against the Langfuse baseline. Exits non-zero on failure.

---

## CI Pipeline

Three-level GitHub Actions workflow (`.github/workflows/eval.yml`), triggered manually (`workflow_dispatch`):

| Level | Trigger | What runs |
|-------|---------|-----------|
| 1 — Assertions | Always | `grep` checks + `pytest tests/unit/` |
| 2 — LLM Eval | `agent/**` or `prompts/**` changed | `eval/run_eval.py` + PR comment with score |
| 3 — Trajectory | `agent/**` or `tests/**` changed | `pytest tests/evals/` |

**Required GitHub Secrets**: `GEMINI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `JUDGE_MODEL`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`

---

## Key Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Gemini LLM + embedding |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `OPENAI_API_BASE` | — | OpenAI-compat endpoint (activates Mode 2) |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model (768-dim) |
| `RAG_TOP_K` | `5` | Chunks to retrieve per query |
| `LANGFUSE_SECRET_KEY` | — | From Langfuse UI → Settings → API Keys |
| `LANGFUSE_PUBLIC_KEY` | — | From Langfuse UI → Settings → API Keys |
| `LANGFUSE_HOST` | `http://langfuse:3000` | Internal Docker address |
| `LANGFUSE_ENVIRONMENT` | `production` | Prompt label to fetch |
| `DATABASE_URL` | `postgresql://langfuse:langfuse@postgres:5432/langfuse` | Shared DB |

---

## Project Structure

```
.
├── agent/              # ADK agent (model selection, Langfuse callbacks, Kafka consumer)
│   └── tools/          # embed.py (dual-mode), retrieve.py (pgvector RAG)
├── api/                # FastAPI REST + SSE + DB helpers
├── docs/               # Insurance knowledge base (Markdown)
├── eval/               # LLM-as-judge eval (dataset, judge, runner)
├── frontend/           # Single-page chat UI
├── infra/              # OTel Collector, Prometheus, Tempo, Grafana configs
├── tests/
│   ├── unit/           # Layer 1: offline unit tests
│   └── evals/          # Layer 2: trajectory tests (real LLM, mock tool)
├── ingest.py           # One-time knowledge base ingestion script
├── docker-compose.yml          # App stack
├── docker-compose.infra.yml    # Infra stack (Langfuse, Grafana, OTel, Postgres)
├── Dockerfile
└── up.sh               # One-command startup
```

---

## Known Issues

- **ADK `tool_trajectory_avg_score` always 0.0**: ADK bug [#5410](https://github.com/google/adk-python/issues/5410). Fix in PR #5417 (pending merge). Trajectory tests use `InMemoryRunner` as workaround.
- **Langfuse OTel integration disabled**: `LANGFUSE_TRACING_ENABLED=false` is required because the Python SDK v4+ tries to export OTel to a v3 server endpoint (`/api/public/otel`) which doesn't exist in the self-hosted Langfuse v2 image. LLM traces are instead recorded via explicit `lf.trace()` callbacks in `agent/agent.py`.
- **`gemini-2.5-flash-lite` tool call reliability**: The lite model's free tier rate limits cause ~50% failure on the second LLM call after a tool use. Use `gemini-2.5-flash` instead.
