from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import sys

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
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


def aggregate_counts(paths: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    structural_counter: Counter[str] = Counter()
    semantic_counter: Counter[str] = Counter()

    for record in iter_records(paths):
        dialogue_type = record.get("dialogue_type", "<unknown>")
        for turn in record.get("turn_labels", []):
            structural_label = turn.get("structural_label", "<missing>")
            semantic_label = turn.get("semantic_label") or "<未标注>"

            structural_counter[(structural_label, dialogue_type)] += 1
            semantic_counter[(semantic_label, dialogue_type)] += 1

    structural_df = pd.DataFrame(
        (
            {"structural_label": label, "dialogue_type": dtype, "count": count}
            for (label, dtype), count in structural_counter.items()
        )
    )
    semantic_df = pd.DataFrame(
        (
            {"semantic_label": label, "dialogue_type": dtype, "count": count}
            for (label, dtype), count in semantic_counter.items()
        )
    )
    return structural_df, semantic_df


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

    structural_df, semantic_df = aggregate_counts(labeled_files)

    plot_distribution(structural_df, "structural_label", args.figure_dir / "structural_distribution.png")
    plot_distribution(semantic_df, "semantic_label", args.figure_dir / "semantic_distribution.png")

    save_summary(structural_df, args.summary_dir / "structural_distribution.csv")
    save_summary(semantic_df, args.summary_dir / "semantic_distribution.csv")


if __name__ == "__main__":
    main()
