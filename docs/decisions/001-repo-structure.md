# 001 Repo Structure

## Decision

Organize the repo as a lab with runnable examples and a reusable Python package.

## Why

The learning path includes separate topics: tool calling, Markdown parsing, Vespa retrieval, LangGraph orchestration, MCP packaging, and evaluation. Keeping each topic as an example makes it easier to learn one layer at a time.

## Consequences

- Examples can be small and runnable.
- Mature code can move into `src/agentic_rag_lab`.
- Tests can target reusable modules directly.

