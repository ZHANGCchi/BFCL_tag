import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, TextIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from convert_oai_to_sgpt import convert_multiturn_to_multi_sample


def normalize_tag(tag: str) -> str:
    """Normalize tag string so it can be safely used as part of a filename."""

    replacements = {
        "<": "",
        ">": "",
        " ": "_",
        "/": "-",
        "|": "-",
        "\\": "-",
    }
    normalized = tag
    for src, tgt in replacements.items():
        normalized = normalized.replace(src, tgt)
    return normalized


def ensure_handle(handles: Dict[Path, TextIO], path: Path) -> TextIO:
    if path not in handles:
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[path] = path.open("w", encoding="utf8")
    return handles[path]


def gather_tags(turn_labels: Iterable[dict]) -> Dict[str, set]:
    buckets = defaultdict(set)
    for label in turn_labels:
        structural = label.get("structural_label")
        semantic = label.get("semantic_label")
        if structural:
            buckets["structural"].add(structural)
        if semantic:
            buckets["semantic"].add(semantic)
    return buckets


def process_file(
    file_path: Path,
    raw_dir: Path,
    sgpt_dir: Path,
    tag_types: Iterable[str],
    raw_handles: Dict[Path, TextIO],
    sgpt_handles: Dict[Path, TextIO],
    stats: Dict[str, Dict[str, int]],
) -> None:
    with file_path.open("r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            tag_map = gather_tags(data.get("turn_labels", []))
            if not tag_map:
                continue

            sgpt_samples = convert_multiturn_to_multi_sample(data)
            for tag_type in tag_types:
                labels = tag_map.get(tag_type)
                if not labels:
                    continue
                for label in labels:
                    normalized = normalize_tag(label)

                    raw_path = raw_dir / tag_type / f"{normalized}.jsonl"
                    raw_handle = ensure_handle(raw_handles, raw_path)
                    raw_handle.write(json.dumps(data, ensure_ascii=False) + "\n")

                    if sgpt_samples:
                        sgpt_path = sgpt_dir / tag_type / f"{normalized}.jsonl"
                        sgpt_handle = ensure_handle(sgpt_handles, sgpt_path)
                        for sample in sgpt_samples:
                            sgpt_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    stats[tag_type][label] = stats[tag_type].get(label, 0) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Split labeled OpenAI-style datasets by tag and convert to SGPT format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/labeled"),
        help="Directory containing labeled JSONL files.",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=Path("output/tag_split/raw"),
        help="Directory to store raw conversations split by tag.",
    )
    parser.add_argument(
        "--sgpt-output-dir",
        type=Path,
        default=Path("output/tag_split/sgpt"),
        help="Directory to store SGPT-formatted conversations split by tag.",
    )
    parser.add_argument(
        "--tag-types",
        nargs="+",
        choices=["structural", "semantic"],
        default=["structural", "semantic"],
        help="Tag types to split on.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    raw_handles: Dict[Path, TextIO] = {}
    sgpt_handles: Dict[Path, TextIO] = {}
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    try:
        for path in sorted(args.input_dir.rglob("*.jsonl")):
            process_file(
                file_path=path,
                raw_dir=args.raw_output_dir,
                sgpt_dir=args.sgpt_output_dir,
                tag_types=args.tag_types,
                raw_handles=raw_handles,
                sgpt_handles=sgpt_handles,
                stats=stats,
            )
    finally:
        for handle in raw_handles.values():
            handle.close()
        for handle in sgpt_handles.values():
            handle.close()

    summary_path = args.sgpt_output_dir.parent / "tag_stats.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Raw tag-split datasets written to: {args.raw_output_dir}")
    print(f"SGPT tag-split datasets written to: {args.sgpt_output_dir}")
    print(f"Tag statistics saved to: {summary_path}")


if __name__ == "__main__":
    main()