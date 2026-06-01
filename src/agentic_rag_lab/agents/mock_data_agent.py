"""OpenAI-compatible function-calling orchestration loop for the mock data agent."""

from __future__ import annotations

import json
import os
from typing import Any

from agentic_rag_lab.core.logging import AgentLogger, compact_preview
from agentic_rag_lab.examples_data.mock_orders import RAW_DATASETS
from agentic_rag_lab.tools.mock_data_tools import TOOL_SPECS, run_tool, validate_answer


SYSTEM_INSTRUCTION = """You are a data-analysis agent demo.

Follow these rules:
1. Use dataset_id=demo_orders_q2 unless the user explicitly provides another valid dataset id.
2. Do not invent numbers. If the answer needs computed values, call a tool first.
3. Before the final answer, call validate_answer and include key numbers in claimed_metrics.
4. In the final answer, write in Traditional Chinese and include a short list of tool actions, findings, and validation status.
5. Do not reveal hidden chain-of-thought. A brief decision summary is fine.
"""

TOOLS = [{"type": "function", "function": spec} for spec in TOOL_SPECS]


class VllmDataAgent:
    """Wrapper around an OpenAI-compatible endpoint with function calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        log_path: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "not-needed")
        self.model = model or os.getenv("VLLM_MODEL", "gemma-4-31B-it")
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://your-vllm-endpoint:port/v1")
        self.log = AgentLogger(log_path or os.getenv("AGENT_LOG_PATH", "logs/agent_run.jsonl"))
        self._client = None

    def _load_sdk(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def run(self, instruction: str, dataset_id: str = "demo_orders_q2", max_steps: int = 8) -> str:
        client = self._load_sdk()

        raw_data = RAW_DATASETS[dataset_id]
        prompt = (
            f"User instruction: {instruction}\n\n"
            f"Raw data dataset_id={dataset_id}:\n{raw_data}\n\n"
            "Choose the tools you need. After tool-based analysis, validate the key numbers before answering."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        validation_seen = False

        self.log.event(
            "run_started",
            dataset_id=dataset_id,
            instruction=instruction,
            raw_row_count=len(raw_data),
            model=self.model,
        )

        for step in range(1, max_steps + 1):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                final_text = message.content or ""
                self.log.event("model_final", step=step, char_count=len(final_text))
                if not validation_seen:
                    guardrail = validate_answer(dataset_id=dataset_id, claimed_metrics={})
                    self.log.event(
                        "guardrail_validation",
                        note="Model did not call validate_answer before final response; local deterministic validation was run with no claimed metrics.",
                        validation=guardrail["validation"],
                        preview=compact_preview(guardrail),
                    )
                return final_text

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                self.log.event(
                    "model_requested_tool",
                    step=step,
                    tool_name=name,
                    args=args,
                    decision_summary=f"Model selected {name} based on the user instruction and tool declaration.",
                )
                result = run_tool(name, args)
                if name == "validate_answer":
                    validation_seen = True
                self.log.event(
                    "tool_finished",
                    tool_name=name,
                    validation=result.get("validation", {}),
                    preview=compact_preview(result),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })

        guardrail = validate_answer(dataset_id=dataset_id, claimed_metrics={})
        self.log.event(
            "guardrail_validation",
            note=f"Reached max_steps={max_steps}; stopping with deterministic validation snapshot.",
            validation=guardrail["validation"],
            preview=compact_preview(guardrail),
        )
        raise RuntimeError(f"Agent reached max_steps={max_steps} without a final answer.")


# Backward-compatible alias
GeminiDataAgent = VllmDataAgent
