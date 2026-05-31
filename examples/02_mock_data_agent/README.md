# 02 Mock Data Agent

This is the current working demo.

It demonstrates a small Gemini agent that can choose among local mock tools:

- `inspect_schema`
- `clean_orders`
- `compute_kpis`
- `detect_risk_signals`
- `validate_answer`

Run:

```bash
uv run mock-data-agent
```

Or run the example wrapper:

```bash
uv run python examples/02_mock_data_agent/run.py --instruction "Find revenue, risks, and validate the numbers."
```

Reusable code lives in:

```text
src/agentic_rag_lab/agents/mock_data_agent.py
src/agentic_rag_lab/tools/mock_data_tools.py
src/agentic_rag_lab/examples_data/mock_orders.py
```
