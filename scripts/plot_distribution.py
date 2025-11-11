from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import sys

import matplotlib
from matplotlib import font_manager
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running the script directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.io_utils import list_jsonl_files
from src.labeling.turns import split_turns
def setup_fonts(font_path: str | None = None, font_family: str | None = None) -> None:
    """Ensure a CJK-capable font is available so Chinese labels render correctly.

    Priority:
    1) Use explicit --font-path if provided
    2) Try common CJK font families if present
    3) Try known system font file paths (Linux)
    4) Fall back to DejaVu Sans (may not cover CJK fully)
    """

    tried: list[str] = []

    def _apply_family(name: str) -> bool:
        try:
            prop = font_manager.FontProperties(family=[name])
            font_manager.findfont(prop, fallback_to_default=False)
            matplotlib.rcParams['font.sans-serif'] = [name]
            return True
        except Exception:
            return False

    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            # Use the family name encoded in the font file
            family_name = prop.get_name()
            if family_name and _apply_family(family_name):
                return
        except Exception:
            tried.append(f"file:{font_path}")

    if font_family:
        if _apply_family(font_family):
            return
        tried.append(f"family:{font_family}")

    common_families = [
        # Common open fonts
        'Noto Sans CJK SC', 'Noto Sans CJK', 'Noto Sans SC',
        'Source Han Sans CN', 'Source Han Sans SC', 'Source Han Sans',
        'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti',
    ]
    for name in common_families:
        if _apply_family(name):
            return
        tried.append(f"family:{name}")

    # Try common Linux font files
    common_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    for p in common_paths:
        path = Path(p)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                prop = font_manager.FontProperties(fname=str(path))
                fam = prop.get_name()
                if fam and _apply_family(fam):
                    return
            except Exception:
                tried.append(f"file:{p}")

    # Fallback to DejaVu Sans so plots still render, with a warning in title
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # Note: We avoid raising to keep non-CJK environments working.



def iter_records(paths: Iterable[Path]) -> Iterator[Dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def aggregate_counts(paths: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, dict]:
    """Aggregate counts across all files and also produce per-file summaries.

    Returns (
        structural_df,
        semantic_df,
        combo_df,                 # 结构×语义 组合分布（所有 turn）
        combo_available_df,       # 结构×语义 组合分布（仅可用 turn：turn 内含 assistant(loss=True)）
        per_file_summaries,
        overall_summary,
    )
    """
    overall_structural: Counter[str] = Counter()
    overall_semantic: Counter[str] = Counter()
    overall_combo: Counter[tuple[str, str, str]] = Counter()
    overall_combo_available: Counter[tuple[str, str, str]] = Counter()
    per_file_summaries: list = []

    for path in paths:
        file_structural: Counter[str] = Counter()
        file_semantic: Counter[str] = Counter()
        file_combo: Counter[tuple[str, str, str]] = Counter()
        file_combo_available: Counter[tuple[str, str, str]] = Counter()
        record_count = 0
        turn_count = 0

        for record in iter_records([path]):
            record_count += 1
            dialogue_type = record.get("dialogue_type", "<unknown>")
            messages = record.get("messages") or []
            # 对应 turn_labels 的 turn 消息切分（与标注阶段一致）
            turns_msgs = split_turns(messages)

            for turn in record.get("turn_labels", []):
                turn_count += 1
                structural_label = turn.get("structural_label", "<missing>")
                semantic_label = turn.get("semantic_label") or "<未标注>"
                t_idx = turn.get("turn_index")

                key_s = (structural_label, dialogue_type)
                key_m = (semantic_label, dialogue_type)
                file_structural[key_s] += 1
                file_semantic[key_m] += 1

                overall_structural[key_s] += 1
                overall_semantic[key_m] += 1

                # 组合计数（所有 turn）
                combo_key = (structural_label, semantic_label, dialogue_type)
                file_combo[combo_key] += 1
                overall_combo[combo_key] += 1

                # 可用样本：该 turn 内是否存在 assistant 且 loss=True
                is_available = False
                if isinstance(t_idx, int) and 0 <= t_idx < len(turns_msgs):
                    for msg in turns_msgs[t_idx]:
                        if msg.get("role") == "assistant" and msg.get("loss") is True:
                            is_available = True
                            break
                if is_available:
                    file_combo_available[combo_key] += 1
                    overall_combo_available[combo_key] += 1

        # Convert tuple keys to string keys for JSON/CSV serializability.
        file_structural_str = {f"{label}|{dtype}": count for (label, dtype), count in file_structural.items()}
        file_semantic_str = {f"{label}|{dtype}": count for (label, dtype), count in file_semantic.items()}

        per_file_summaries.append(
            {
                "file_name": path.name,
                "path": str(path),
                "record_count": record_count,
                "turn_count": turn_count,
                "structural_counts": file_structural_str,
                "semantic_counts": file_semantic_str,
                # 组合统计写为字符串键，便于持久化
                "combo_counts": {f"{s}|{m}|{dtype}": c for (s, m, dtype), c in file_combo.items()},
                "combo_available_counts": {f"{s}|{m}|{dtype}": c for (s, m, dtype), c in file_combo_available.items()},
            }
        )

    structural_df = pd.DataFrame(
        (
            {"structural_label": label, "dialogue_type": dtype, "count": count}
            for (label, dtype), count in overall_structural.items()
        )
    )
    semantic_df = pd.DataFrame(
        (
            {"semantic_label": label, "dialogue_type": dtype, "count": count}
            for (label, dtype), count in overall_semantic.items()
        )
    )

    combo_df = pd.DataFrame(
        (
            {
                "structural_label": s,
                "semantic_label": m,
                "dialogue_type": dtype,
                "count": count,
            }
            for (s, m, dtype), count in overall_combo.items()
        )
    )

    combo_available_df = pd.DataFrame(
        (
            {
                "structural_label": s,
                "semantic_label": m,
                "dialogue_type": dtype,
                "count": count,
            }
            for (s, m, dtype), count in overall_combo_available.items()
        )
    )

    # Prepare overall counts dicts with string keys for serialization.
    overall_structural_str = {f"{label}|{dtype}": count for (label, dtype), count in overall_structural.items()}
    overall_semantic_str = {f"{label}|{dtype}": count for (label, dtype), count in overall_semantic.items()}

    overall_summary = {
        "record_count": sum(p["record_count"] for p in per_file_summaries),
        "turn_count": sum(p["turn_count"] for p in per_file_summaries),
        "structural_counts": overall_structural_str,
        "semantic_counts": overall_semantic_str,
        "combo_counts": {f"{s}|{m}|{dtype}": c for (s, m, dtype), c in overall_combo.items()},
        "combo_available_counts": {f"{s}|{m}|{dtype}": c for (s, m, dtype), c in overall_combo_available.items()},
    }

    return structural_df, semantic_df, combo_df, combo_available_df, per_file_summaries, overall_summary


def plot_distribution(df: pd.DataFrame, label_column: str, output_path: Path) -> None:
    if df.empty:
        raise ValueError("No data available to plot.")

    pivot = df.pivot_table(index=label_column, columns="dialogue_type", values="count", fill_value=0)
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_xlabel("标签")
    ax.set_ylabel("对话轮次数量")
    ax.set_title(f"{label_column} 分布（按对话类型分组）")
    ax.legend(title="对话类型")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combo_distribution(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """绘制 结构×语义 的组合分布（按对话类型堆叠柱状）。"""
    if df.empty:
        raise ValueError("No data available to plot (combo).")
    df = df.copy()
    df["combo_label"] = df["structural_label"].astype(str) + " × " + df["semantic_label"].astype(str)
    pivot = df.pivot_table(index="combo_label", columns="dialogue_type", values="count", fill_value=0)
    # 排序：按总量降序
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(by="__total__", ascending=False).drop(columns=["__total__"])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(14, 7))
    ax.set_xlabel("结构 × 语义 组合")
    ax.set_ylabel("对话轮次数量")
    ax.set_title(title)
    ax.legend(title="对话类型")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_distribution_flat(df: pd.DataFrame, label_column: str, output_path: Path, title: str) -> None:
    """不区分对话类型的总分布柱状图。"""
    if df.empty:
        raise ValueError("No data available to plot (flat).")
    pivot = df.groupby(label_column, as_index=False)["count"].sum().sort_values("count", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(pivot[label_column], pivot["count"]) 
    plt.xlabel("标签")
    plt.ylabel("对话轮次数量")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combo_distribution_flat(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """不区分对话类型的 结构×语义 组合分布。"""
    if df.empty:
        raise ValueError("No data available to plot (combo flat).")
    tmp = df.copy()
    tmp["combo_label"] = tmp["structural_label"].astype(str) + " × " + tmp["semantic_label"].astype(str)
    pivot = tmp.groupby("combo_label", as_index=False)["count"].sum().sort_values("count", ascending=False)
    plt.figure(figsize=(14, 7))
    plt.bar(pivot["combo_label"], pivot["count"]) 
    plt.xlabel("结构 × 语义 组合")
    plt.ylabel("对话轮次数量")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot label distributions from labeled datasets.")
    parser.add_argument("--labeled-dir", type=Path, required=True, help="Directory containing labeled .jsonl files.")
    parser.add_argument("--figure-dir", type=Path, required=True, help="Directory to store generated figures.")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="Directory to store summary CSV files.",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        help="Optional absolute path to a CJK-capable font file (e.g., NotoSansCJK-Regular.ttc).",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        help="Optional font family name to use (e.g., 'Noto Sans CJK SC', 'WenQuanYi Zen Hei').",
    )
    parser.add_argument(
        "--no-dialogue-type",
        action="store_true",
        help="Do not split by dialogue type (Single/Multi); aggregate counts across types.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure a CJK-capable font is configured so Chinese text displays in figures
    setup_fonts(font_path=args.font_path, font_family=args.font_family)
    labeled_files = list_jsonl_files(args.labeled_dir)
    if not labeled_files:
        raise FileNotFoundError(f"No labeled .jsonl files found in {args.labeled_dir}")

    (
        structural_df,
        semantic_df,
        combo_df,
        combo_available_df,
        per_file_summaries,
        overall_summary,
    ) = aggregate_counts(labeled_files)

    if args.no_dialogue_type:
        # 不区分对话类型的聚合与绘图
        plot_distribution_flat(structural_df, "structural_label", args.figure_dir / "structural_distribution_all.png", "结构标签分布（不区分对话类型）")
        plot_distribution_flat(semantic_df, "semantic_label", args.figure_dir / "semantic_distribution_all.png", "语义标签分布（不区分对话类型）")
        plot_combo_distribution_flat(combo_df, args.figure_dir / "combo_distribution_all.png", "结构×语义 组合分布（不区分对话类型）")
        plot_combo_distribution_flat(combo_available_df, args.figure_dir / "combo_available_distribution_all.png", "结构×语义 组合分布（可用 Turn，不区分对话类型）")

        # 保存聚合后的 CSV（合并各类型）
        structural_flat = structural_df.groupby("structural_label", as_index=False)["count"].sum()
        semantic_flat = semantic_df.groupby("semantic_label", as_index=False)["count"].sum()
        combo_flat = combo_df.groupby(["structural_label", "semantic_label"], as_index=False)["count"].sum()
        combo_avail_flat = combo_available_df.groupby(["structural_label", "semantic_label"], as_index=False)["count"].sum()
        save_summary(structural_flat, args.summary_dir / "structural_distribution_all.csv")
        save_summary(semantic_flat, args.summary_dir / "semantic_distribution_all.csv")
        save_summary(combo_flat, args.summary_dir / "combo_distribution_all.csv")
        save_summary(combo_avail_flat, args.summary_dir / "combo_available_distribution_all.csv")
    else:
        # 按对话类型分组的原有绘图/导出
        plot_distribution(structural_df, "structural_label", args.figure_dir / "structural_distribution.png")
        plot_distribution(semantic_df, "semantic_label", args.figure_dir / "semantic_distribution.png")
        plot_combo_distribution(combo_df, args.figure_dir / "combo_distribution.png", "结构×语义 组合分布（所有 Turn，按对话类型分组）")
        plot_combo_distribution(combo_available_df, args.figure_dir / "combo_available_distribution.png", "结构×语义 组合分布（可用 Turn：含 loss=True 的 assistant）")

        save_summary(structural_df, args.summary_dir / "structural_distribution.csv")
        save_summary(semantic_df, args.summary_dir / "semantic_distribution.csv")
        save_summary(combo_df, args.summary_dir / "combo_distribution.csv")
        save_summary(combo_available_df, args.summary_dir / "combo_available_distribution.csv")

    # Save per-file summaries and overall summary
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    per_file_rows = []
    for info in per_file_summaries:
        per_file_rows.append(
            {
                "file_name": info["file_name"],
                "path": info["path"],
                "record_count": info["record_count"],
                "turn_count": info["turn_count"],
                "structural_counts": json.dumps(info["structural_counts"], ensure_ascii=False),
                "semantic_counts": json.dumps(info["semantic_counts"], ensure_ascii=False),
                "combo_counts": json.dumps(info.get("combo_counts", {}), ensure_ascii=False),
                "combo_available_counts": json.dumps(info.get("combo_available_counts", {}), ensure_ascii=False),
            }
        )

    per_file_df = pd.DataFrame(per_file_rows)
    per_file_df.to_csv(args.summary_dir / "per_file_summary.csv", index=False, encoding="utf-8-sig")

    # overall summary to JSON and a flattened CSV
    with (args.summary_dir / "overall_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(overall_summary, fh, ensure_ascii=False, indent=2)

    overall_flat = {
        "record_count": overall_summary["record_count"],
        "turn_count": overall_summary["turn_count"],
        "structural_counts": json.dumps(overall_summary["structural_counts"], ensure_ascii=False),
        "semantic_counts": json.dumps(overall_summary["semantic_counts"], ensure_ascii=False),
        "combo_counts": json.dumps(overall_summary.get("combo_counts", {}), ensure_ascii=False),
        "combo_available_counts": json.dumps(overall_summary.get("combo_available_counts", {}), ensure_ascii=False),
    }
    pd.DataFrame([overall_flat]).to_csv(args.summary_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
