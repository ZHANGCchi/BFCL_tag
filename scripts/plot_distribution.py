from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import sys

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running the script directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling.io_utils import list_jsonl_files


def iter_records(paths: Iterable[Path]) -> Iterator[Dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def aggregate_counts(paths: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Aggregate counts across all files and also produce per-file summaries.

    Returns (structural_df, semantic_df, per_file_summaries, overall_summary)
    """
    overall_structural: Counter[str] = Counter()
    overall_semantic: Counter[str] = Counter()
    per_file_summaries: list = []

    for path in paths:
        file_structural: Counter[str] = Counter()
        file_semantic: Counter[str] = Counter()
        record_count = 0
        turn_count = 0

        for record in iter_records([path]):
            record_count += 1
            dialogue_type = record.get("dialogue_type", "<unknown>")
            for turn in record.get("turn_labels", []):
                turn_count += 1
                structural_label = turn.get("structural_label", "<missing>")
                semantic_label = turn.get("semantic_label") or "<未标注>"

                key_s = (structural_label, dialogue_type)
                key_m = (semantic_label, dialogue_type)
                file_structural[key_s] += 1
                file_semantic[key_m] += 1

                overall_structural[key_s] += 1
                overall_semantic[key_m] += 1

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

    # Prepare overall counts dicts with string keys for serialization.
    overall_structural_str = {f"{label}|{dtype}": count for (label, dtype), count in overall_structural.items()}
    overall_semantic_str = {f"{label}|{dtype}": count for (label, dtype), count in overall_semantic.items()}

    overall_summary = {
        "record_count": sum(p["record_count"] for p in per_file_summaries),
        "turn_count": sum(p["turn_count"] for p in per_file_summaries),
        "structural_counts": overall_structural_str,
        "semantic_counts": overall_semantic_str,
    }

    return structural_df, semantic_df, per_file_summaries, overall_summary


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labeled_files = list_jsonl_files(args.labeled_dir)
    if not labeled_files:
        raise FileNotFoundError(f"No labeled .jsonl files found in {args.labeled_dir}")

    structural_df, semantic_df, per_file_summaries, overall_summary = aggregate_counts(labeled_files)

    plot_distribution(structural_df, "structural_label", args.figure_dir / "structural_distribution.png")
    plot_distribution(semantic_df, "semantic_label", args.figure_dir / "semantic_distribution.png")

    save_summary(structural_df, args.summary_dir / "structural_distribution.csv")
    save_summary(semantic_df, args.summary_dir / "semantic_distribution.csv")

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
    }
    pd.DataFrame([overall_flat]).to_csv(args.summary_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
