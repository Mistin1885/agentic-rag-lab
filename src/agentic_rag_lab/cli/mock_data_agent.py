"""Command-line entrypoint for the vLLM / OpenAI-compatible agent demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_rag_lab.agents.mock_data_agent import VllmDataAgent
from agentic_rag_lab.examples_data.mock_orders import DEFAULT_INSTRUCTION


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Run the OpenAI-compatible function-calling agent demo.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="User instruction for the agent.")
    parser.add_argument("--dataset-id", default="demo_orders_q2", help="Dataset id to analyze.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to VLLM_MODEL env var.")
    parser.add_argument("--base-url", default=None, help="API base URL. Defaults to VLLM_BASE_URL env var.")
    parser.add_argument("--log-path", default=None, help="JSONL log path. Defaults to AGENT_LOG_PATH or logs/agent_run.jsonl.")
    args = parser.parse_args()

    agent = VllmDataAgent(model=args.model, base_url=args.base_url, log_path=args.log_path)
    final = agent.run(instruction=args.instruction, dataset_id=args.dataset_id)

    print("\n=== Final Answer ===")
    print(final)
    print("\n=== Log File ===")
    print(Path(agent.log.path).resolve())


if __name__ == "__main__":
    main()
