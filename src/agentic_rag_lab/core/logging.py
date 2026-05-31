"""Readable console and JSONL logging for the demo agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AgentLogger:
    """Write compact action logs to console and JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(self._format_console(record), flush=True)

    def _format_console(self, record: dict[str, Any]) -> str:
        event = record["event"]
        if event == "run_started":
            return f"[agent] start dataset={record['dataset_id']} instruction={record['instruction']}"
        if event == "model_requested_tool":
            return (
                f"[agent] tool_call step={record['step']} name={record['tool_name']} "
                f"args={json.dumps(record['args'], ensure_ascii=False)}"
            )
        if event == "tool_finished":
            ok = record.get("validation", {}).get("ok")
            return f"[tool] done name={record['tool_name']} validation_ok={ok} preview={record['preview']}"
        if event == "model_final":
            return f"[agent] final step={record['step']} chars={record['char_count']}"
        if event == "guardrail_validation":
            ok = record.get("validation", {}).get("ok")
            return f"[guardrail] validation_ok={ok} note={record['note']}"
        return f"[{event}] {json.dumps(record, ensure_ascii=False)}"


def compact_preview(value: Any, limit: int = 500) -> str:
    """Return a short JSON preview safe for console display."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

