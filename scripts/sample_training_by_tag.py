#!/usr/bin/env python
"""Sample labeled datasets to build a training set with controlled tag proportions."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from convert_oai_to_sgpt import convert_multiturn_to_multi_sample  # noqa: E402
from src.labeling.turns import split_turns  # noqa: E402

SUPPORTED_TAG_TYPES = ("structural", "semantic")
DEFAULT_LABELS = {"structural": "<UNLABELED_STRUCTURAL>", "semantic": "<NO_SEMANTIC>"}


@dataclass
class SampleRecord:
    raw_sample: dict
    labels: Tuple[str, ...]
    source_id: str
    turn_index: int


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf8") as f:
        return json.load(f)


def classify_targets(values: Dict[str, float]) -> str:
    if not values:
        return "none"
    if all(isinstance(v, (int, float)) and v <= 1 for v in values.values()):
        return "proportion"
    return "count"


def normalize_counts_to_total(raw_counts: Dict[str, float], total: int) -> Dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not raw_counts:
        return {}
    current_sum = sum(raw_counts.values())
    if current_sum == 0:
        raise ValueError("Cannot normalize zero-sum counts")
    scale = total / current_sum
    scaled = {k: raw_counts[k] * scale for k in raw_counts}
    base = {k: int(math.floor(v)) for k, v in scaled.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        fractions = sorted(
            ((scaled[k] - base[k], k) for k in scaled),
            reverse=True,
        )
        idx = 0
        while remainder > 0 and idx < len(fractions):
            _, key = fractions[idx]
            base[key] += 1
            remainder -= 1
            idx += 1
        if remainder > 0:
            keys = [key for _, key in fractions]
            while remainder > 0 and keys:
                for key in keys:
                    base[key] += 1
                    remainder -= 1
                    if remainder == 0:
                        break
    elif remainder < 0:
        keys = sorted(base, key=lambda k: base[k], reverse=True)
        idx = 0
        while remainder < 0 and keys:
            key = keys[idx % len(keys)]
            if base[key] > 0:
                base[key] -= 1
                remainder += 1
            idx += 1
            if idx > len(keys) * 10:
                break
    return base


def proportions_to_counts(proportions: Dict[str, float], total: int) -> Dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not proportions:
        return {}
    positive = {k: max(0.0, float(v)) for k, v in proportions.items()}
    total_prop = sum(positive.values())
    if total_prop <= 0:
        raise ValueError("Proportions sum must be > 0")
    normalized = {k: v / total_prop for k, v in positive.items()}
    scaled = {k: normalized[k] * total for k in normalized}
    base = {k: int(math.floor(v)) for k, v in scaled.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        fractions = sorted(
            ((scaled[k] - base[k], k) for k in scaled),
            reverse=True,
        )
        idx = 0
        while remainder > 0 and idx < len(fractions):
            _, key = fractions[idx]
            base[key] += 1
            remainder -= 1
            idx += 1
        if remainder > 0:
            keys = [key for _, key in fractions]
            while remainder > 0 and keys:
                for key in keys:
                    base[key] += 1
                    remainder -= 1
                    if remainder == 0:
                        break
    elif remainder < 0:
        keys = sorted(base, key=lambda k: base[k], reverse=True)
        idx = 0
        while remainder < 0 and keys:
            key = keys[idx % len(keys)]
            if base[key] > 0:
                base[key] -= 1
                remainder += 1
            idx += 1
            if idx > len(keys) * 10:
                break
    return base


def compute_target_counts(
    config: Dict[str, object],
    tag_types: Sequence[str],
    total_samples_override: Optional[int],
) -> Tuple[Dict[str, Dict[str, int]], int]:
    raw_total = config.get("total_samples")
    total_samples: Optional[int] = total_samples_override or raw_total

    specs = {}
    count_totals = []

    for tag_type in tag_types:
        raw_values = config.get(tag_type) or {}
        mode = classify_targets(raw_values)
        specs[tag_type] = {"mode": mode, "values": raw_values}
        if mode == "count" and raw_values:
            count_totals.append(sum(int(round(v)) for v in raw_values.values()))

    if total_samples is None:
        if count_totals:
            total_samples = count_totals[0]
            for tot in count_totals[1:]:
                if tot != total_samples:
                    raise ValueError(
                        "Count targets across tag types must sum to the same total."
                    )
        else:
            raise ValueError(
                "total_samples is required when all targets are expressed as proportions."
            )

    total_samples = int(total_samples)
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")

    targets: Dict[str, Dict[str, int]] = {}
    for tag_type in tag_types:
        spec = specs[tag_type]
        mode = spec["mode"]
        raw_values = spec["values"]
        if mode == "none" or not raw_values:
            targets[tag_type] = {}
            continue
        if mode == "proportion":
            counts = proportions_to_counts({k: float(v) for k, v in raw_values.items()}, total_samples)
        else:
            counts = normalize_counts_to_total({k: float(v) for k, v in raw_values.items()}, total_samples)
        targets[tag_type] = counts

    # Ensure totals are consistent
    non_empty_totals = [sum(counts.values()) for counts in targets.values() if counts]
    for tot in non_empty_totals:
        if tot != total_samples:
            raise ValueError(
                f"Target counts for tag types must sum to total_samples ({total_samples})."
            )

    return targets, total_samples


def extract_labels_for_turn(turn_label: dict, tag_types: Sequence[str]) -> Optional[Tuple[str, ...]]:
    labels: List[str] = []
    for tag_type in tag_types:
        if tag_type == "structural":
            value = turn_label.get("structural_label")
            if not value:
                return None
        elif tag_type == "semantic":
            value = turn_label.get("semantic_label") or DEFAULT_LABELS[tag_type]
        else:
            raise ValueError(f"Unsupported tag type: {tag_type}")
        labels.append(value)
    return tuple(labels)


def extract_turn_samples(data: dict) -> Dict[int, Dict[str, Optional[dict]]]:
    """
    提取 turn 样本。
    
    关键逻辑：
    1. 使用 split_turns 按 user 消息划分 turn（与标注逻辑一致）
    2. 对于 turn_index=N 的样本，messages 包含从开始到 turnN 结束的所有消息
    3. 这样可以为 SGPT 提供完整上下文，但需要在转换时避免重复
    """
    messages = data.get("messages") or []
    if not messages:
        return {}

    tools = data.get("tools")
    samples: Dict[int, Dict[str, Optional[dict]]] = {}
    
    # 使用与标注相同的逻辑划分 turn
    turns = split_turns(messages)
    
    # 累积消息：构建每个 turn 的完整上下文
    accumulated_messages = []
    
    for turn_index, turn_messages in enumerate(turns):
        # 添加当前 turn 的消息到累积列表
        accumulated_messages.extend(turn_messages)
        
        # 检查当前 turn 是否有 loss=True 的 assistant 消息
        # 如果没有，说明这个 turn 不需要训练，跳过
        has_loss_true = any(
            msg.get("role") == "assistant" and msg.get("loss") == True
            for msg in turn_messages
        )
        
        if not has_loss_true:
            continue
        
        # 深拷贝并清理
        raw_messages = copy.deepcopy(accumulated_messages)
        for _msg in raw_messages:
            if _msg.get("content"):
                _msg["content"] = _msg["content"].strip()
            # 保留 reasoning_content，因为转换时需要
        
        sample_id = f"{data.get('id', '')}_turn_{turn_index}"
        
        raw_sample = {
            "id": sample_id,
            "source_id": data.get("id"),
            "turn_index": turn_index,
            "messages": raw_messages,
            "tools": tools,
        }
        
        samples[turn_index] = {"raw": raw_sample}
    
    return samples


def build_sample_index(
    labeled_dir: Path,
    tag_types: Sequence[str],
    rng: random.Random,
    max_files: Optional[int] = None,
) -> Tuple[
    Dict[Tuple[str, ...], List[SampleRecord]],
    Dict[str, Dict[str, int]],
    Dict[str, dict],
    Dict[str, Dict[int, Tuple[str, ...]]],
]:
    samples_by_combo: Dict[Tuple[str, ...], List[SampleRecord]] = defaultdict(list)
    availability: Dict[str, Dict[str, int]] = {
        tag_type: defaultdict(int) for tag_type in tag_types
    }
    conversations: Dict[str, dict] = {}
    labels_by_conversation: Dict[str, Dict[int, Tuple[str, ...]]] = defaultdict(dict)

    files: Iterable[Path]
    files = sorted([p for p in labeled_dir.glob("*.jsonl") if p.is_file()])
    if max_files is not None:
        files = files[:max_files]

    for file_path in files:
        with file_path.open("r", encoding="utf8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                conv_id = data.get("id")
                if not conv_id:
                    conv_id = f"{file_path.stem}:{line_idx}"
                    data["id"] = conv_id
                conversations[conv_id] = data
                turn_labels = data.get("turn_labels") or []
                if not turn_labels:
                    continue
                labels_by_turn = {lbl.get("turn_index"): lbl for lbl in turn_labels}
                turn_samples = extract_turn_samples(data)
                if not turn_samples:
                    continue

                for turn_index, label_info in labels_by_turn.items():
                    if turn_index is None or turn_index < 0:
                        continue
                    sample_pair = turn_samples.get(turn_index)
                    if not sample_pair:
                        continue

                    labels = extract_labels_for_turn(label_info, tag_types)
                    if not labels:
                        continue

                    raw_sample = copy.deepcopy(sample_pair["raw"])
                    raw_sample["dialogue_type"] = data.get("dialogue_type")
                    selected_tags = {
                        tag_type: labels[idx]
                        for idx, tag_type in enumerate(tag_types)
                    }
                    raw_sample["selected_tags"] = selected_tags

                    for idx, tag_type in enumerate(tag_types):
                        availability[tag_type][labels[idx]] += 1
                    labels_by_conversation[conv_id][turn_index] = labels

                    record = SampleRecord(
                        raw_sample=raw_sample,
                        labels=labels,
                        source_id=data.get("id", ""),
                        turn_index=turn_index,
                    )
                    samples_by_combo[labels].append(record)

    for combo_records in samples_by_combo.values():
        rng.shuffle(combo_records)

    return samples_by_combo, availability, conversations, labels_by_conversation


def allocate_single_dimension(
    tag_type: str,
    target_counts: Dict[str, int],
    samples_by_combo: Dict[Tuple[str, ...], List[SampleRecord]],
    allow_shortfall: bool,
) -> Tuple[List[SampleRecord], Dict[str, int], Dict[str, int]]:
    selected: List[SampleRecord] = []
    actual: Dict[str, int] = defaultdict(int)
    shortfall: Dict[str, int] = {}

    for label, required in target_counts.items():
        combo_key = (label,)
        combo_samples = samples_by_combo.get(combo_key, [])
        available = len(combo_samples)
        take = min(required, available)
        if take > 0:
            selected.extend(combo_samples[:take])
            actual[label] += take
        if take < required:
            missing = required - take
            shortfall[label] = missing
            if not allow_shortfall:
                raise ValueError(
                    f"Insufficient samples for {tag_type}='{label}': "
                    f"required {required}, available {available}."
                )
    return selected, actual, shortfall


def allocate_two_dimensions(
    tag_types: Sequence[str],
    target_counts: Dict[str, Dict[str, int]],
    samples_by_combo: Dict[Tuple[str, ...], List[SampleRecord]],
    allow_shortfall: bool,
) -> Tuple[List[SampleRecord], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    primary_type, secondary_type = tag_types
    primary_targets = target_counts.get(primary_type, {})
    secondary_targets = target_counts.get(secondary_type, {})

    allocation: Dict[Tuple[str, ...], int] = defaultdict(int)
    available_counts = {combo: len(records) for combo, records in samples_by_combo.items()}

    remaining_primary = primary_targets.copy()
    remaining_secondary = secondary_targets.copy()

    combos_by_primary: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    combos_by_secondary: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

    for combo in samples_by_combo:
        primary_label, secondary_label = combo
        combos_by_primary[primary_label].append(combo)
        combos_by_secondary[secondary_label].append(combo)

    def available_for_combo(combo: Tuple[str, ...]) -> int:
        return available_counts.get(combo, 0) - allocation.get(combo, 0)

    def viable_combos_for_primary(label: str) -> List[Tuple[str, ...]]:
        combos = []
        for combo in combos_by_primary.get(label, []):
            if available_for_combo(combo) <= 0:
                continue
            secondary_label = combo[1]
            if remaining_secondary.get(secondary_label, 0) <= 0:
                continue
            combos.append(combo)
        return combos

    def viable_combos_for_secondary(label: str) -> List[Tuple[str, ...]]:
        combos = []
        for combo in combos_by_secondary.get(label, []):
            if available_for_combo(combo) <= 0:
                continue
            primary_label = combo[0]
            if remaining_primary.get(primary_label, 0) <= 0:
                continue
            combos.append(combo)
        return combos

    changed = True
    while changed:
        changed = False
        for label, remaining in list(remaining_primary.items()):
            if remaining <= 0:
                continue
            combos = viable_combos_for_primary(label)
            if len(combos) == 1:
                combo = combos[0]
                capacity = min(
                    remaining,
                    remaining_secondary.get(combo[1], 0),
                    available_for_combo(combo),
                )
                if capacity > 0:
                    allocation[combo] += capacity
                    remaining_primary[label] -= capacity
                    remaining_secondary[combo[1]] -= capacity
                    changed = True
        for label, remaining in list(remaining_secondary.items()):
            if remaining <= 0:
                continue
            combos = viable_combos_for_secondary(label)
            if len(combos) == 1:
                combo = combos[0]
                capacity = min(
                    remaining,
                    remaining_primary.get(combo[0], 0),
                    available_for_combo(combo),
                )
                if capacity > 0:
                    allocation[combo] += capacity
                    remaining_secondary[label] -= capacity
                    remaining_primary[combo[0]] -= capacity
                    changed = True

    while True:
        candidates = []
        for combo, total_available in available_counts.items():
            remaining_avail = total_available - allocation.get(combo, 0)
            if remaining_avail <= 0:
                continue
            primary_label, secondary_label = combo
            if remaining_primary.get(primary_label, 0) <= 0:
                continue
            if remaining_secondary.get(secondary_label, 0) <= 0:
                continue
            primary_options = len(viable_combos_for_primary(primary_label))
            secondary_options = len(viable_combos_for_secondary(secondary_label))
            capacity = min(
                remaining_avail,
                remaining_primary[primary_label],
                remaining_secondary[secondary_label],
            )
            if capacity <= 0:
                continue
            candidates.append(
                ((primary_options, secondary_options, remaining_avail), combo, capacity)
            )
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0][0], item[0][1], item[0][2]))
        _, combo, capacity = candidates[0]
        allocation[combo] += capacity
        remaining_primary[combo[0]] -= capacity
        remaining_secondary[combo[1]] -= capacity

    shortfall_primary = {
        label: remaining
        for label, remaining in remaining_primary.items()
        if remaining > 0
    }
    shortfall_secondary = {
        label: remaining
        for label, remaining in remaining_secondary.items()
        if remaining > 0
    }

    if (shortfall_primary or shortfall_secondary) and not allow_shortfall:
        messages = []
        if shortfall_primary:
            messages.append(
                ", ".join(
                    f"{primary_type}='{label}' short by {missing}"
                    for label, missing in shortfall_primary.items()
                )
            )
        if shortfall_secondary:
            messages.append(
                ", ".join(
                    f"{secondary_type}='{label}' short by {missing}"
                    for label, missing in shortfall_secondary.items()
                )
            )
        raise ValueError("; ".join(messages))

    selected: List[SampleRecord] = []
    for combo, count in allocation.items():
        if count <= 0:
            continue
        combo_samples = samples_by_combo.get(combo, [])
        selected.extend(combo_samples[:count])

    actual_counts: Dict[str, Dict[str, int]] = {
        primary_type: defaultdict(int),
        secondary_type: defaultdict(int),
    }
    for record in selected:
        actual_counts[primary_type][record.labels[0]] += 1
        actual_counts[secondary_type][record.labels[1]] += 1

    shortfall = {
        primary_type: shortfall_primary,
        secondary_type: shortfall_secondary,
    }

    actual = {
        primary_type: dict(actual_counts[primary_type]),
        secondary_type: dict(actual_counts[secondary_type]),
    }

    return selected, actual, shortfall


def allocate_samples(
    tag_types: Sequence[str],
    target_counts: Dict[str, Dict[str, int]],
    samples_by_combo: Dict[Tuple[str, ...], List[SampleRecord]],
    allow_shortfall: bool,
) -> Tuple[List[SampleRecord], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    if not tag_types:
        raise ValueError("At least one tag type must be specified")
    if len(tag_types) == 1:
        tag_type = tag_types[0]
        selected, actual, shortfall = allocate_single_dimension(
            tag_type,
            target_counts.get(tag_type, {}),
            samples_by_combo,
            allow_shortfall,
        )
        return selected, {tag_type: actual}, {tag_type: shortfall}
    if len(tag_types) == 2:
        return allocate_two_dimensions(
            tag_types,
            target_counts,
            samples_by_combo,
            allow_shortfall,
        )
    raise NotImplementedError("Only up to two tag types are currently supported")


def summarize_selected(
    selected: Sequence[SampleRecord],
    tag_types: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {
        tag_type: defaultdict(int) for tag_type in tag_types
    }
    for record in selected:
        for idx, tag_type in enumerate(tag_types):
            summary[tag_type][record.labels[idx]] += 1
    return {tag_type: dict(counts) for tag_type, counts in summary.items()}


def save_raw_jsonl(path: Path, samples: Sequence[SampleRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf8") as f:
        for record in samples:
            f.write(json.dumps(record.raw_sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_turn_index(sample_id: Optional[str]) -> Optional[int]:
    if not sample_id:
        return None
    if "_turn_" not in sample_id:
        return None
    try:
        return int(sample_id.rsplit("_turn_", 1)[1])
    except ValueError:
        return None


def save_sgpt_jsonl(
    path: Path,
    conversations: Dict[str, dict],
    selected_turns: Dict[str, Set[int]],
    raw_samples: Dict[Tuple[str, int], dict],  # (source_id, turn_index) -> raw_sample
    raw_id_lookup: Dict[Tuple[str, int], str],  # 用于判断哪些样本是真正选中的
) -> Tuple[int, int, Dict[str, List[dict]]]:
    """
    转换并保存 SGPT 样本。
    
    关键逻辑：
    - 对每个选中的 turn，使用其 raw_sample 进行转换
    - raw_sample 包含了从对话开始到该 turn 结束的所有消息（提供完整上下文）
    - convert_multiturn_to_multi_sample 会为 raw_sample 中所有 loss=True 的 assistant 生成 SGPT 样本
    - **问题**：这包括之前 turn 的 assistant（上下文中的），不只是当前 turn 的
    - **解决**：根据 raw_id_lookup 过滤，只保留真正属于选中 turn 的样本
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    total_count = 0
    selected_count = 0
    sgpt_by_conversation: Dict[str, List[dict]] = {}
    
    def _count_loss_true_in_turn(turn_messages: List[dict]) -> int:
        return sum(1 for m in turn_messages if m.get("role") == "assistant" and m.get("loss") is True)

    def _allowed_counter_range_for_last_turn(raw_messages: List[dict]) -> Tuple[int, int]:
        """
        给定原始消息（已按 turnN 累积到末尾），计算只属于最后一个 turn 的样本计数区间 [start, end]（含端点）。

        convert_multiturn_to_multi_sample 会遍历 raw_messages，遇到 assistant(loss=True) 就按序号生成样本：
        第一个命中 -> _turn_0, 第二个 -> _turn_1, ...

        我们要保留“最后一个 turn 内”的所有命中，因此：
        - offset = 之前所有 turn 中的 loss=True 的数量
        - count_last = 最后一个 turn 中的 loss=True 的数量
        - 合法计数范围 = [offset, offset + count_last - 1]
        若 count_last 为 0，则返回空区间 (1, 0)。
        """
        turns_local = split_turns(raw_messages)
        if not turns_local:
            return (1, 0)
        if len(turns_local) == 1:
            start = 0
            end = _count_loss_true_in_turn(turns_local[0]) - 1
            return (start, end)

        offset = 0
        for t_msgs in turns_local[:-1]:
            offset += _count_loss_true_in_turn(t_msgs)
        last_cnt = _count_loss_true_in_turn(turns_local[-1])
        if last_cnt <= 0:
            return (1, 0)
        return (offset, offset + last_cnt - 1)

    def _parse_last_counter(sample_id: str) -> Optional[int]:
        """从类似 "API:1003_turn_2_turn_1" 中解析最后一个 _turn_ 后的计数 1。"""
        if "_turn_" not in sample_id:
            return None
        try:
            return int(sample_id.rsplit("_turn_", 1)[-1])
        except Exception:
            return None

    with path.open("w", encoding="utf8") as f:
        for source_id in sorted(selected_turns.keys()):
            selected_indices = selected_turns[source_id]
            all_sgpt_for_conv = []
            
            for turn_idx in sorted(selected_indices):
                # 获取该 turn 的 raw sample
                raw_key = (source_id, turn_idx)
                raw_sample = raw_samples.get(raw_key)
                if not raw_sample:
                    continue
                
                # 转换该 raw sample（包含完整上下文）
                sgpt_samples = convert_multiturn_to_multi_sample(raw_sample)
                
                # 过滤：只保留真正属于“当前选中 turn（最后一个 turn 段）”的样本。
                # 依据计数区间 [start,end]（见上文说明）筛选 sample_id 末尾的 counter。
                start, end = _allowed_counter_range_for_last_turn(raw_sample.get("messages") or [])
                filtered_samples = []
                if start <= end:
                    raw_id_expect = raw_sample.get("id")
                    for sample in sgpt_samples:
                        sample_id = sample.get("id", "")
                        # 可选的安全检查：确保属于当前 raw_id
                        if raw_id_expect and sample_id.startswith(str(raw_id_expect)):
                            counter = _parse_last_counter(sample_id)
                            if counter is not None and start <= counter <= end:
                                filtered_samples.append(sample)
                                selected_count += 1
                
                all_sgpt_for_conv.extend(filtered_samples)
                
                # 只写入过滤后的样本
                serialized = [json.dumps(sample, ensure_ascii=False) + "\n" for sample in filtered_samples]
                f.writelines(serialized)
                total_count += len(filtered_samples)
            
            sgpt_by_conversation[source_id] = all_sgpt_for_conv
    
    return total_count, selected_count, sgpt_by_conversation


def save_metadata(
    path: Path,
    tag_types: Sequence[str],
    sgpt_samples: Dict[str, List[dict]],
    labels_by_conversation: Dict[str, Dict[int, Tuple[str, ...]]],
    raw_id_lookup: Dict[Tuple[str, int], str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for source_id in sorted(sgpt_samples.keys()):
            label_map = labels_by_conversation.get(source_id, {})
            for sample in sgpt_samples[source_id]:
                sample_id = sample.get("id")
                turn_index = extract_turn_index(sample_id)
                meta: Dict[str, object] = {
                    "source_id": source_id,
                    "sgpt_id": sample_id,
                    "turn_index": turn_index,
                }
                raw_key = (source_id, turn_index)
                if raw_key in raw_id_lookup:
                    meta["raw_id"] = raw_id_lookup[raw_key]
                    meta["selected_for_sampling"] = True
                else:
                    meta["raw_id"] = None
                    meta["selected_for_sampling"] = False
                labels = label_map.get(turn_index)
                if labels:
                    for idx, tag_type in enumerate(tag_types):
                        meta[tag_type] = labels[idx]
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample SGPT training data with controlled tag distributions.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON file specifying target distributions.",
    )
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=Path("output/labeled"),
        help="Directory containing labeled JSONL files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/training/training_dataset.jsonl"),
        help="Path to write the sampled SGPT dataset.",
    )
    parser.add_argument(
        "--raw-output-jsonl",
        type=Path,
        default=Path("output/training/raw/selected.jsonl"),
        help="Path to write the sampled raw turn dataset.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/training/sample_report.json"),
        help="Path to write allocation summary report.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional path to write metadata for selected samples.",
    )
    parser.add_argument(
        "--tag-types",
        nargs="+",
        choices=SUPPORTED_TAG_TYPES,
        default=list(SUPPORTED_TAG_TYPES),
        help="Tag types to balance (default: structural semantic).",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        help="Override total sample size (overrides config total_samples).",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow the script to proceed even if targets cannot be fully met.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on the number of labeled files to scan (for testing).",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    rng = random.Random(args.seed)

    target_counts, total_samples = compute_target_counts(
        config,
        args.tag_types,
        args.total_samples,
    )

    (
        samples_by_combo,
        availability,
        conversations,
        labels_by_conversation,
    ) = build_sample_index(
        args.labeled_dir,
        args.tag_types,
        rng,
        max_files=args.max_files,
    )

    selected_samples, actual_counts, shortfall = allocate_samples(
        args.tag_types,
        target_counts,
        samples_by_combo,
        allow_shortfall=args.allow_shortfall,
    )

    rng.shuffle(selected_samples)

    raw_selected = save_raw_jsonl(args.raw_output_jsonl, selected_samples)

    selected_turns: Dict[str, Set[int]] = defaultdict(set)
    raw_id_lookup: Dict[Tuple[str, int], str] = {}
    raw_samples: Dict[Tuple[str, int], dict] = {}  # (source_id, turn_index) -> raw_sample
    
    for record in selected_samples:
        selected_turns[record.source_id].add(record.turn_index)
        raw_id = record.raw_sample.get("id")
        if raw_id:
            raw_id_lookup[(record.source_id, record.turn_index)] = raw_id
        # 保存 raw_sample 用于转换
        raw_samples[(record.source_id, record.turn_index)] = record.raw_sample

    sgpt_total, sgpt_selected, sgpt_samples = save_sgpt_jsonl(
        args.output_jsonl,
        conversations,
        selected_turns,
        raw_samples,
        raw_id_lookup,  # 传入 raw_id_lookup 用于过滤
    )

    if args.metadata_output:
        save_metadata(
            args.metadata_output,
            args.tag_types,
            sgpt_samples,
            labels_by_conversation,
            raw_id_lookup,
        )

    summary = summarize_selected(selected_samples, args.tag_types)
    selected_turn_total = sum(len(turns) for turns in selected_turns.values())

    report = {
        "config": {
            "total_samples": total_samples,
            "tag_types": args.tag_types,
            "targets": target_counts,
        },
        "selection": {
            "unit": "turn",
            "total_selected": len(selected_samples),
            "unique_turns": selected_turn_total,
            "raw_selected": raw_selected,
            "sgpt_total": sgpt_total,
            "sgpt_selected": sgpt_selected,
            "actual": summary,
            "shortfall": shortfall,
        },
        "availability": {
            tag_type: dict(counts) for tag_type, counts in availability.items()
        },
        "notes": [
            "total_selected/raw_selected 表示按标签采样得到的 turn 数量",
            "sgpt_total 和 sgpt_selected 应该相等,均表示对采样 turn 转换得到的 SGPT 样本数量"
        ],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Target total (turns): {total_samples}")
    print(f"Selected turns: {len(selected_samples)}")
    print(f"Unique turn ids: {selected_turn_total}")
    print(f"Raw turns written: {raw_selected}")
    print(f"SGPT samples generated: {sgpt_total}")
    if sgpt_total != sgpt_selected:
        print(f"WARNING: sgpt_total ({sgpt_total}) != sgpt_selected ({sgpt_selected})")
    for tag_type in args.tag_types:
        print(f"\n[{tag_type}]")
        targets = target_counts.get(tag_type, {})
        actual = summary.get(tag_type, {})
        short = shortfall.get(tag_type, {})
        for label in sorted(set(list(targets.keys()) + list(actual.keys()))):
            t = targets.get(label, 0)
            a = actual.get(label, 0)
            s = short.get(label, 0)
            print(f"  {label}: target={t}, actual={a}, shortfall={s}")


if __name__ == "__main__":
    main()
