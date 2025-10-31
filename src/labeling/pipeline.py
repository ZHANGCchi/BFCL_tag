from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from .io_utils import count_jsonl_rows, ensure_output_dir, list_jsonl_files, read_jsonl, write_jsonl
from .structural import analyse_structural_label
from .turns import detect_dialogue_type, split_turns

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnLabel:
    turn_index: int
    structural_label: str
    semantic_label: Optional[str]
    total_calls: int
    unique_tool_count: int
    tool_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "structural_label": self.structural_label,
            "semantic_label": self.semantic_label,
            "total_calls": self.total_calls,
            "unique_tool_count": self.unique_tool_count,
            "tool_names": self.tool_names,
        }


class LabelingPipeline:
    """Pipeline that attaches structural and semantic turn labels to datasets."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        semantic_judge_factory: Callable[[], Any],
        show_turn_progress: bool = True,
        resume: bool = True,
    ) -> None:
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._semantic_judge_factory = semantic_judge_factory
        self._show_turn_progress = show_turn_progress
        # If resume is True, files that already exist in the output directory
        # (same filename) will be skipped to allow resuming a previously
        # interrupted run.
        self._resume = resume

    def _iter_dialogues(self) -> Iterator[Path]:
        return iter(list_jsonl_files(self._data_dir))

    def _process_dialogue(
        self,
        source_path: Path,
        semantic_judge: Any,
        position: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        processed_records: List[Dict[str, Any]] = []
        structural_counter: Counter[tuple[str, str]] = Counter()
        semantic_counter: Counter[tuple[str, str]] = Counter()
        total_turns = 0

        progress_bar: Optional[tqdm] = None
        if self._show_turn_progress:
            total_rows = count_jsonl_rows(source_path)
            progress_bar = tqdm(
                total=total_rows or None,
                desc=source_path.name,
                unit="record",
                leave=False,
                position=position,
            )

        for record in read_jsonl(source_path):
            messages = record.get("messages") or []
            tools = record.get("tools") or []

            dialogue_type = detect_dialogue_type(messages)
            turns = split_turns(messages)

            labeled_turns: List[TurnLabel] = []

            for index, turn in enumerate(turns):
                structural_info = analyse_structural_label(turn, tools)
                context = {
                    "file": str(source_path),
                    "record_index": len(processed_records),
                    "turn_index": index,
                }
                semantic_label = semantic_judge.judge(turn, tools, dialogue_type, context=context)

                labeled_turns.append(
                    TurnLabel(
                        turn_index=index,
                        structural_label=structural_info["label"],
                        semantic_label=semantic_label,
                        total_calls=structural_info["total_calls"],
                        unique_tool_count=structural_info["unique_tool_count"],
                        tool_names=structural_info["tool_names"],
                    )
                )

                structural_counter[(structural_info["label"], dialogue_type)] += 1
                semantic_counter[((semantic_label or "<未标注>"), dialogue_type)] += 1
                total_turns += 1

            record["dialogue_type"] = dialogue_type
            record["turn_labels"] = [turn_label.to_dict() for turn_label in labeled_turns]
            processed_records.append(record)
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        file_summary = {
            "record_count": len(processed_records),
            "turn_count": total_turns,
            "structural_counts": dict(structural_counter),
            "semantic_counts": dict(semantic_counter),
        }
        return processed_records, file_summary

    def _process_file(self, source_path: Path, position: Optional[int]) -> Dict[str, Any]:
        semantic_judge = self._semantic_judge_factory()
        processed_records, summary = self._process_dialogue(source_path, semantic_judge, position=position)
        target_path = self._output_dir / source_path.name
        ensure_output_dir(target_path)
        write_jsonl(target_path, processed_records)
        summary["file_name"] = source_path.name
        summary["path"] = str(source_path)
        return summary

    def run(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        ensure_output_dir(self._output_dir)
        source_paths = list(self._iter_dialogues())
        # If resume is enabled, skip input files that already have an output
        # file present in the output directory. This implements a simple
        # "skip-existing" resume behavior.
        if self._resume:
            kept: List[Path] = []
            skipped_files: List[str] = []
            for p in source_paths:
                target = self._output_dir / p.name
                if target.exists():
                    skipped_files.append(p.name)
                else:
                    kept.append(p)
            if skipped_files:
                logger.info("Skipping %d files because outputs exist: %s", len(skipped_files), ", ".join(skipped_files[:10]))
            source_paths = kept
        if not source_paths:
            return {"files": [], "overall": {"record_count": 0, "turn_count": 0, "structural_counts": {}, "semantic_counts": {}}}

        summaries: List[Dict[str, Any]] = []
        overall_structural: Counter[tuple[str, str]] = Counter()
        overall_semantic: Counter[tuple[str, str]] = Counter()
        total_records = 0
        total_turns = 0

        positions = {path: idx + 1 for idx, path in enumerate(source_paths)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_file, path, positions[path] if self._show_turn_progress else None): path
                for path in source_paths
            }
            with tqdm(total=len(futures), desc="Labeling files", unit="file") as progress:
                for future in as_completed(futures):
                    summary = future.result()
                    summaries.append(summary)
                    total_records += summary["record_count"]
                    total_turns += summary["turn_count"]
                    overall_structural.update(summary["structural_counts"])
                    overall_semantic.update(summary["semantic_counts"])
                    progress.update(1)

        overall = {
            "record_count": total_records,
            "turn_count": total_turns,
            "structural_counts": dict(overall_structural),
            "semantic_counts": dict(overall_semantic),
        }
        summaries.sort(key=lambda item: item["file_name"])
        return {"files": summaries, "overall": overall}
