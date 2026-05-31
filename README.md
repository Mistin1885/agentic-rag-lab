# Agentic RAG Lab

This repo is a learning lab for building up from small tool-calling agents to agentic RAG systems with LangGraph, Vespa, and MCP.

The current working demo is `examples/02_mock_data_agent`, migrated from the original `src/agent_demo` prototype.

## Current Layout

```text
agentic-rag-lab/
  docs/                         Learning notes and design decisions
  data/                         Raw and processed practice datasets
  examples/                     Small runnable exercises
  src/agentic_rag_lab/          Reusable Python package
  tests/                        Unit and integration tests
  scripts/                      Ingestion, indexing, and evaluation scripts
  infra/                        Local service configs such as Vespa
```

## Examples

```text
examples/01_tool_calling_basic   Planned minimal function-calling exercise
examples/02_mock_data_agent      Current Gemini + mock tools demo
examples/03_markdown_parser      Planned Markdown parsing exercise
```

## Current Demo

The mock data agent shows:

- Gemini function calling
- Local mock tools
- Tool selection from user instruction
- Deterministic validation
- JSONL action logs

Install dependencies with `uv`, then set `GOOGLE_API_KEY` in `.env`:

```bash
uv sync --extra dev
cp .env.example .env
uv run mock-data-agent
```

Or:

```bash
uv run agent-demo --instruction "Find Q2 revenue, risky customers, and validate the final metrics."
```

Logs are written to:

```text
logs/agent_run.jsonl
```

## Test

The deterministic mock tools can be tested without a Gemini API key:

```bash
uv run python -m unittest discover -s tests -v
```

## Learning Roadmap

See [docs/roadmap.md](docs/roadmap.md).
