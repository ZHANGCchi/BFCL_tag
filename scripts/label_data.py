from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running the script directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen3 import Qwen3

from src.labeling.pipeline import LabelingPipeline
from src.labeling.semantic import SemanticJudge


class SemanticLLMClient:
    """Thin wrapper that prompts Qwen3 for semantic labels."""

    def __init__(self, max_retries: int = 2) -> None:
        self._model = Qwen3(max_retries=max_retries)

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        turn_messages = payload["messages"]
        tools = payload["available_tools"]
        assistant_content = payload["assistant_content"]

        instruction = {
            "task": "Identify if the assistant response complains about missing parameters or missing tools.",
            "output_format": {
                "type": "json",
                "schema": {
                    "missing_parameters": "bool",
                    "missing_tools": "bool",
                },
            },
            "criteria": {
                "missing_parameters": "Assistant explicitly says it lacks required inputs, fields, arguments, slots, ids, or parameters to complete the task.",
                "missing_tools": "Assistant explicitly says the available toolset lacks a required capability or mentions not having a suitable/required tool.",
                "otherwise": "Return false for both flags.",
            },
        }

        prompt = (
            "你是一名标注员，需要根据对话判断助手机器人的最后一条自然语言回复是否表示缺少参数或缺少工具。"
            "请严格按照下述JSON格式输出，不要包含额外文字。\n"
            f"对话turn: {json.dumps(turn_messages, ensure_ascii=False)}\n"
            f"可用工具: {json.dumps(tools, ensure_ascii=False)}\n"
            f"需要评估的回复: {json.dumps(assistant_content, ensure_ascii=False)}\n"
            f"标注说明: {json.dumps(instruction, ensure_ascii=False)}\n"
            "输出:"
        )

        raw = self._model(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM返回结果无法解析为JSON: {raw}") from exc

        if not {"missing_parameters", "missing_tools"}.issubset(result):
            raise ValueError(f"LLM返回结果缺少必要字段: {result}")

        return {
            "missing_parameters": bool(result.get("missing_parameters")),
            "missing_tools": bool(result.get("missing_tools")),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label dialogue datasets with turn-level metadata.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Input directory that contains .jsonl files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where labeled jsonl files will be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    semantic_client = SemanticLLMClient()
    semantic_judge = SemanticJudge(client=semantic_client)

    pipeline = LabelingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        semantic_judge=semantic_judge,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
