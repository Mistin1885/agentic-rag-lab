# Repo Map

Use `examples/` for learning chapters and `src/agentic_rag_lab/` for reusable code.

## Rules of Thumb

- Markdown parsing should not require an LLM.
- Retrieval should not require an agent.
- Tools should wrap stable local capabilities.
- LangGraph should orchestrate tools, not hide retrieval logic.
- MCP should expose existing tools, not become the only place business logic lives.

