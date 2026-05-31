"""Gemini function-calling orchestration loop for the mock data agent."""

from __future__ import annotations

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


class GeminiDataAgent:
    """Small wrapper around Gemini generate_content with function calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        log_path: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.log = AgentLogger(log_path or os.getenv("AGENT_LOG_PATH", "logs/agent_run.jsonl"))
        self._client = None
        self._types = None

    def _load_sdk(self) -> tuple[Any, Any]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is required. Copy .env.example to .env and set your key.")
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self.api_key)
            self._types = types
        return self._client, self._types

    def run(self, instruction: str, dataset_id: str = "demo_orders_q2", max_steps: int = 8) -> str:
        client, types = self._load_sdk()
        tool_config = types.Tool(function_declarations=TOOL_SPECS)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[tool_config],
            temperature=0.2,
        )

        raw_data = RAW_DATASETS[dataset_id]
        prompt = (
            f"User instruction: {instruction}\n\n"
            f"Raw data dataset_id={dataset_id}:\n{raw_data}\n\n"
            "Choose the tools you need. After tool-based analysis, validate the key numbers before answering."
        )
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        validation_seen = False

        self.log.event(
            "run_started",
            dataset_id=dataset_id,
            instruction=instruction,
            raw_row_count=len(raw_data),
            model=self.model,
        )

        for step in range(1, max_steps + 1):
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            function_calls = list(getattr(response, "function_calls", None) or [])

            if not function_calls:
                final_text = getattr(response, "text", "") or ""
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

            contents.append(response.candidates[0].content)

            for call in function_calls:
                args = dict(call.args or {})
                self.log.event(
                    "model_requested_tool",
                    step=step,
                    tool_name=call.name,
                    args=args,
                    decision_summary=f"Model selected {call.name} based on the user instruction and tool declaration.",
                )
                result = run_tool(call.name, args)
                if call.name == "validate_answer":
                    validation_seen = True
                self.log.event(
                    "tool_finished",
                    tool_name=call.name,
                    validation=result.get("validation", {}),
                    preview=compact_preview(result),
                )
                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name,
                                response=result,
                            )
                        ],
                    )
                )

        guardrail = validate_answer(dataset_id=dataset_id, claimed_metrics={})
        self.log.event(
            "guardrail_validation",
            note=f"Reached max_steps={max_steps}; stopping with deterministic validation snapshot.",
            validation=guardrail["validation"],
            preview=compact_preview(guardrail),
        )
        raise RuntimeError(f"Agent reached max_steps={max_steps} without a final answer.")
