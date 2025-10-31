from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

JsonDict = Dict[str, Any]


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    """Yield JSON objects from a `.jsonl` file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    """Write iterable of JSON objects to a `.jsonl` file."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_output_dir(path: Path) -> None:
    """Create parent directory for a path if it does not exist."""
    target_dir = path if not path.suffix else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)


def list_jsonl_files(folder: Path) -> List[Path]:
    """Return all `.jsonl` files within a directory."""
    return sorted(folder.glob("*.jsonl"))


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty lines in a `.jsonl` file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
