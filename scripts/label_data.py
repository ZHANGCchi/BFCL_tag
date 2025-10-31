from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running the script directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen3 import Qwen3

from src.labeling.pipeline import LabelingPipeline
from src.labeling.semantic import SemanticJudge


_LOGGER_SETUP_LOCK = Lock()


class SemanticLLMClient:
    """Thin wrapper that prompts Qwen3 for semantic labels."""

    def __init__(self, max_retries: int = 2, log_path: Optional[Path] = None) -> None:
        self._model = Qwen3(max_retries=max_retries)
        self._logger = logging.getLogger("semantic_llm")
        if log_path:
            with _LOGGER_SETUP_LOCK:
                if not self._logger.handlers:
                    log_file = log_path if log_path.is_absolute() else (PROJECT_ROOT / log_path)
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    handler = logging.FileHandler(log_file, encoding="utf-8")
                    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                    self._logger.addHandler(handler)
                    self._logger.setLevel(logging.INFO)
                    self._logger.propagate = False
        self._log_enabled = bool(log_path)

    def _log_event(self, event: str, context: Optional[Dict[str, Any]], content: str) -> None:
        if not self._log_enabled or not self._logger.handlers:
            return
        payload = {
            "event": event,
            "context": context or {},
            "content": content,
        }
        self._logger.info(json.dumps(payload, ensure_ascii=False))

    def __call__(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            f"需要评估的回复: {json.dumps(assistant_content, ensure_ascii=False)}\n"
            f"标注说明: {json.dumps(instruction, ensure_ascii=False)}\n"
            "输出:"
        )

        self._log_event("request", context, prompt)
        raw = self._model(prompt)
        self._log_event("response", context, raw)
        # 提取 JSON 内容，去除 markdown 包裹
        def extract_json(text: str) -> str:
            text = text.strip()
            if text.startswith("```json"):
                text = text[len("```json"):].strip()
            if text.startswith("```"):
                text = text[len("```"):].strip()
            if text.endswith("```"):
                text = text[: -len("```")].strip()
            return text

        json_text = extract_json(raw)
        try:
            result = json.loads(json_text)
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker threads to use for labeling.",
    )
    parser.add_argument(
        "--llm-log-file",
        type=Path,
        default=Path("logs/semantic_llm.log"),
        help="File path to store LLM prompt/response logs.",
    )
    parser.add_argument(
        "--enable-llm-log",
        action="store_true",
        help="Enable logging of LLM prompts and responses to the specified log file.",
    )
    parser.add_argument(
        "--no-turn-progress",
        action="store_true",
        help="Disable per-file record progress bars (useful for non-interactive runs).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def format_counts(counts: Dict[Any, int]) -> str:
    if not counts:
        return "(none)"
    def key_to_str(key: Any) -> str:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            return f"{key[0]}|{key[1]}"
        return str(key)

    parts = [
        f"{key_to_str(label)}={count}"
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    ]
    return ", ".join(parts)


def display_summary(summary: Dict[str, Any]) -> None:
    files = summary.get("files", [])
    overall = summary.get("overall", {})

    print("\nPer-file summary:")
    if not files:
        print("  (no files processed)")
    for info in files:
        print(
            f"- {info['file_name']}: records={info['record_count']}, turns={info['turn_count']}"
        )
        print(f"  structural: {format_counts(info.get('structural_counts', {}))}")
        print(f"  semantic:   {format_counts(info.get('semantic_counts', {}))}")

    print("\nOverall summary:")
    print(
        f"  records={overall.get('record_count', 0)}, turns={overall.get('turn_count', 0)}"
    )
    print(f"  structural: {format_counts(overall.get('structural_counts', {}))}")
    print(f"  semantic:   {format_counts(overall.get('semantic_counts', {}))}\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    llm_log_file = args.llm_log_file
    if args.enable_llm_log:
        if not llm_log_file.is_absolute():
            llm_log_file = (PROJECT_ROOT / llm_log_file).resolve()
    else:
        llm_log_file = None

    def judge_factory() -> SemanticJudge:
        client = SemanticLLMClient(log_path=llm_log_file)
        return SemanticJudge(client=client)

    pipeline = LabelingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        semantic_judge_factory=judge_factory,
        show_turn_progress=not args.no_turn_progress,
    )
    summary = pipeline.run(max_workers=args.max_workers)
    display_summary(summary)


if __name__ == "__main__":
    main()
