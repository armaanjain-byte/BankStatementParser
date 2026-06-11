"""
transaction_reconstructor.py — Row reconstruction from OCR token geometry.

Takes a flat list of OCR tokens (each with a bounding box) and reconstructs
logical table rows by clustering tokens that share the same vertical
position, then sorting left-to-right within each row.

This module does NOT:
  • Classify transactions (debit / credit)
  • Extract dates or amounts
  • Detect table headers vs data rows

It operates purely on geometric coordinates.

Algorithm
─────────
  1. Compute each token's vertical center: ``y_center = (y_min + y_max) / 2``
  2. Sort tokens by ``y_center`` (top → bottom).
  3. Walk through sorted tokens, grouping consecutive tokens whose
     ``y_center`` values differ by ≤ ``threshold`` pixels into the same row.
  4. Within each row, sort tokens left-to-right by ``x_min``.
  5. Concatenate token texts with a single space to produce ``row.text``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from src.schema import OCRToken

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Row dataclass
# ──────────────────────────────────────────────

@dataclass
class Row:
    """A single reconstructed row of text from a bank statement.

    Attributes:
        y_center: Mean vertical center of all tokens in this row (pixels).
        tokens:   The constituent :class:`OCRToken` objects, sorted
                  left-to-right by x-coordinate.
        text:     Concatenated token texts, space-separated. Mutable; modified
                  by downstream normalizers.
        original_text: The original concatenated token texts immediately after
                  reconstruction, before any normalization. IMMUTABLE baseline
                  for recovery detection in evaluation.
    """

    y_center: float
    tokens: List[OCRToken] = field(default_factory=list)
    text: str = ""
    original_text: str = ""

    @property
    def token_count(self) -> int:
        """Number of tokens in this row."""
        return len(self.tokens)

    @property
    def avg_confidence(self) -> float:
        """Mean OCR confidence across tokens in this row."""
        if not self.tokens:
            return 0.0
        return sum(t.confidence for t in self.tokens) / len(self.tokens)

    @property
    def x_min(self) -> int:
        """Leftmost x-coordinate in this row."""
        if not self.tokens:
            return 0
        return min(t.bbox[0] for t in self.tokens)

    @property
    def x_max(self) -> int:
        """Rightmost x-coordinate in this row."""
        if not self.tokens:
            return 0
        return max(t.bbox[2] for t in self.tokens)


# ──────────────────────────────────────────────
#  Core algorithm
# ──────────────────────────────────────────────

def _token_y_center(token: OCRToken) -> float:
    """Compute the vertical center of a token's bounding box."""
    return (token.bbox[1] + token.bbox[3]) / 2.0


def _build_row(tokens: List[OCRToken]) -> Row:
    """Create a Row from a group of tokens.

    Tokens are sorted left-to-right by ``x_min``, then their texts
    are concatenated with a single space separator.
    """
    sorted_tokens = sorted(tokens, key=lambda t: t.bbox[0])
    y_center = sum(_token_y_center(t) for t in sorted_tokens) / len(sorted_tokens)
    text = " ".join(t.text for t in sorted_tokens)

    return Row(
        y_center=round(y_center, 1),
        tokens=sorted_tokens,
        text=text,
        original_text=text,
    )


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def reconstruct_rows(
    tokens: List[OCRToken],
    threshold: float | None = None,
) -> List[Row]:
    """Cluster OCR tokens into logical rows by vertical position.

    Args:
        tokens:    List of :class:`OCRToken` objects from a single page.
        threshold: Optional maximum vertical distance (in pixels) between the
                   y-centers of two tokens for them to be considered
                   part of the same row.  When omitted, this is computed
                   from average token height.

    Returns:
        A list of :class:`Row` objects, ordered top-to-bottom by
        ``y_center``.  Each row's tokens are sorted left-to-right.
    """
    if not tokens:
        return []

    threshold_source = "caller-specified"
    if threshold is None:
        heights = [token.bbox[3] - token.bbox[1] for token in tokens]
        if heights:
            import statistics
            median_height = statistics.median(heights)
            threshold = median_height * 0.5
        else:
            threshold = 6.0
        threshold_source = "adaptive"

    logger.debug(
        "Row clustering threshold: %.1fpx (%s)",
        threshold,
        threshold_source,
    )

    # Step 1: Sort all tokens by vertical center (top → bottom)
    sorted_tokens = sorted(tokens, key=_token_y_center)

    # Step 2: Greedy clustering — walk top-to-bottom, accumulating
    #         tokens into the current group while they stay within
    #         `threshold` of the group's anchor y-center.
    groups: List[List[OCRToken]] = []
    current_group: List[OCRToken] = [sorted_tokens[0]]
    anchor_y = _token_y_center(sorted_tokens[0])

    for token in sorted_tokens[1:]:
        y = _token_y_center(token)

        if abs(y - anchor_y) <= threshold:
            # Same row — add to current group
            current_group.append(token)
            # Update anchor to running mean to handle slight drift
            anchor_y = sum(_token_y_center(t) for t in current_group) / len(current_group)
        else:
            # New row — finalize current group, start a new one
            groups.append(current_group)
            current_group = [token]
            anchor_y = y

    # Don't forget the last group
    groups.append(current_group)

    # Step 3: Build Row objects (sorts tokens left-to-right internally)
    rows = [_build_row(group) for group in groups]

    logger.info(
        "Reconstructed %d rows from %d tokens (threshold=%.1f px)",
        len(rows), len(tokens), threshold,
    )

    return rows


# ──────────────────────────────────────────────
#  JSON serialization
# ──────────────────────────────────────────────

def save_rows_json(
    rows: List[Row],
    output_path: Union[str, Path],
) -> Path:
    """Serialize reconstructed rows to a human-readable JSON file.

    Output schema::

        {
          "row_count": 42,
          "rows": [
            {
              "row_number": 1,
              "y_center": 75.0,
              "token_count": 3,
              "avg_confidence": 0.8123,
              "text": "23/12/2019 UPI Payment 226.14"
            },
            ...
          ]
        }

    Args:
        rows:        List of :class:`Row` objects to serialize.
        output_path: Destination file path.  Parent directories will
                     be created automatically if they don't exist.

    Returns:
        The resolved :class:`Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "row_count": len(rows),
        "rows": [
            {
                "row_number": idx,
                "y_center": row.y_center,
                "token_count": row.token_count,
                "avg_confidence": round(row.avg_confidence, 4),
                "text": row.text,
            }
            for idx, row in enumerate(rows, start=1)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved rows JSON (%d rows) -> %s", len(rows), output_path)
    return output_path


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Reconstruct table rows from an OCR JSON file.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an *_ocr.json file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to outputs/<stem>_rows.json",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Y-clustering threshold in pixels. Defaults to adaptive token-height scaling.",
    )

    args = parser.parse_args()

    # Load OCR tokens
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    ocr_tokens = [
        OCRToken(
            text=t["text"],
            confidence=t["confidence"],
            bbox=tuple(t["bbox"]),
        )
        for t in data["tokens"]
    ]

    if not ocr_tokens:
        logger.warning("No tokens found in %s", args.input)
        sys.exit(0)

    # Reconstruct rows
    rows = reconstruct_rows(ocr_tokens, threshold=args.threshold)

    # Save
    if args.output is None:
        args.output = Path("outputs") / f"{args.input.stem.replace('_ocr', '')}_rows.json"

    save_rows_json(rows, args.output)

    # Print preview
    print(f"\n{'='*70}")
    print(f"Reconstructed {len(rows)} rows from {len(ocr_tokens)} tokens")
    print(f"{'='*70}")
    for row in rows[:20]:
        print(f"  [y={row.y_center:6.1f}] ({row.token_count:2d} tokens, "
              f"conf={row.avg_confidence:.3f})  {row.text[:80]}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more rows")
    print(f"\nSaved -> {args.output}")