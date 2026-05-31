# Roadmap

## Phase 1: Tool Calling

- Build the simplest one-tool agent.
- Build the current mock data agent with multiple tools.
- Add deterministic validation and action logs.

## Phase 2: Markdown Knowledge Base

- Parse Markdown frontmatter, headings, sections, and line ranges.
- Extract metadata such as tags, doc type, customer, date, and status.
- Produce JSONL records for documents, sections, and chunks.

## Phase 3: Retrieval

- Start with keyword and metadata search.
- Add Vespa schema and feed scripts.
- Add BM25, vector search, hybrid search, and ranking profiles.

## Phase 4: Agentic RAG

- Add retrieval tools: metadata search, hybrid section search, read section, read neighbors.
- Add fact extraction, conflict detection, and source validation.
- Add an iterative retrieve-read-validate-answer loop.

## Phase 5: LangGraph

- Convert the agent loop into a graph with explicit state.
- Add conditional edges for insufficient evidence, conflict handling, and clarification.
- Add checkpointing and trace-friendly logs.

## Phase 6: MCP

- Wrap retrieval and validation tools as an MCP server.
- Expose corpus schema and tag taxonomy as MCP resources.
- Keep business logic in `src/agentic_rag_lab`, not inside the MCP adapter.

## Phase 7: Evaluation

- Build a question set with expected answers and expected sources.
- Measure retrieval recall, citation correctness, validation pass rate, latency, and cost.

