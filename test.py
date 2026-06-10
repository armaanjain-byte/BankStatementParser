"""Stdlib smoke test for the end-to-end bank statement parser."""

from __future__ import annotations

import sys
from pathlib import Path

from src.main import IMAGE_EXTENSIONS, process_statement


def main() -> int:
    input_dir = Path("test_set")
    if not input_dir.exists():
        print(f"test_set folder not found: {input_dir}", file=sys.stderr)
        return 1

    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print("No test images found in test_set/", file=sys.stderr)
        return 1

    failures = []
    for image_path in images:
        statement = process_statement(image_path)
        transaction_count = statement.transaction_count
        validation_confidence = statement.validation.confidence
        print(
            f"{image_path.name}: transactions={transaction_count}, "
            f"validation_confidence={validation_confidence:.2f}"
        )
        if transaction_count == 0:
            failures.append(image_path.name)

    if failures:
        print(
            "Images with zero extracted transactions: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
