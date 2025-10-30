from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .io_utils import ensure_output_dir, list_jsonl_files, read_jsonl, write_jsonl
from .semantic import SemanticJudge
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
        semantic_judge: SemanticJudge,
    ) -> None:
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._semantic_judge = semantic_judge

    def _iter_dialogues(self) -> Iterator[Path]:
        return iter(list_jsonl_files(self._data_dir))

    def _process_dialogue(self, source_path: Path) -> Iterator[Dict[str, Any]]:
        for record in read_jsonl(source_path):
            messages = record.get("messages") or []
            tools = record.get("tools") or []

            dialogue_type = detect_dialogue_type(messages)
            turns = split_turns(messages)

            labeled_turns: List[TurnLabel] = []

            for index, turn in enumerate(turns):
                structural_info = analyse_structural_label(turn, tools)
                semantic_label = self._semantic_judge.judge(turn, tools, dialogue_type)

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

            record["dialogue_type"] = dialogue_type
            record["turn_labels"] = [turn_label.to_dict() for turn_label in labeled_turns]
            yield record

    def run(self) -> None:
        ensure_output_dir(self._output_dir)
        for source_path in self._iter_dialogues():
            logger.info("processing file %s", source_path.name)
            target_path = self._output_dir / source_path.name
            processed_records = self._process_dialogue(source_path)
            write_jsonl(target_path, processed_records)
