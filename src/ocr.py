"""
ocr.py — EasyOCR entry point for the Bank Statement Parser.

Responsibilities (and *only* these):
  1. Accept a path to a bank statement image.
  2. Delegate to EasyOCRBackend to extract every text detection.
  3. Convert raw EasyOCR results into a list of OCRToken objects.
  4. Optionally serialize those tokens to a human-readable JSON file.

This module does NOT perform:
  • Field extraction
  • Transaction parsing
  • Any form of validation

Rule 2: Downstream pipeline code must not know which backend produced the
        OCRToken objects.  The public extract_tokens() function here returns
        the same OCRToken contract regardless of backend.

EasyOCR output format (per detection):
    (bbox, text, confidence)
    where bbox is a list of four [x, y] corner points (polygon).
    We convert the polygon to an axis-aligned bounding box
    (x_min, y_min, x_max, y_max) for downstream consumption.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Union

import numpy as np

from src.backends.easyocr_backend import EasyOCRBackend
from src.models.ocr_models import OCRToken

logger = logging.getLogger(__name__)
MIN_CONFIDENCE = 0.10

# Module-level backend singleton (mirrors the original _reader singleton
# pattern so that repeated calls within the same process are cheap).
_backend: EasyOCRBackend = EasyOCRBackend()


# ──────────────────────────────────────────────
#  Core public API
# ──────────────────────────────────────────────

def extract_tokens(
    image_path: Union[str, Path, np.ndarray],
    languages: list[str] | None = None,
    gpu: bool = True,
    preprocess: bool = False,
    min_confidence: float = MIN_CONFIDENCE,
) -> List[OCRToken]:
    """Run OCR on a bank statement image and return structured tokens.

    Args:
        image_path:     Path to the input image (JPEG, PNG, TIFF, etc.)
                        or a NumPy array already loaded into memory.
        languages:      Language codes for EasyOCR (default: ``["en"]``).
        gpu:            Whether to use GPU acceleration.
        preprocess:     Whether to run OpenCV preprocessing before OCR.
        min_confidence: Drop OCR tokens below this confidence score.

    Returns:
        A list of :class:`OCRToken` objects, one per detected text region,
        ordered by EasyOCR's default reading-order heuristic (roughly
        top-to-bottom, left-to-right).

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    return _backend.extract_tokens(
        image_path,
        languages=languages,
        gpu=gpu,
        preprocess=preprocess,
        min_confidence=min_confidence,
    )


# ──────────────────────────────────────────────
#  JSON serialization
# ──────────────────────────────────────────────

def save_ocr_json(
    tokens: List[OCRToken],
    output_path: Union[str, Path],
) -> Path:
    """Serialize a list of OCRTokens to a human-readable JSON file.

    The output schema is:
    ```json
    {
      "token_count": 42,
      "tokens": [
        {
          "text": "HDFC Bank",
          "confidence": 0.9812,
          "bbox": [120, 45, 340, 78]
        },
        ...
      ]
    }
    ```

    Args:
        tokens:      List of :class:`OCRToken` objects to serialize.
        output_path: Destination file path.  Parent directories will
                     be created automatically if they don't exist.

    Returns:
        The resolved :class:`Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "token_count": len(tokens),
        "tokens": [
            {
                "text": t.text,
                "confidence": t.confidence,
                "bbox": list(t.bbox),
            }
            for t in tokens
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved OCR JSON (%d tokens) -> %s", len(tokens), output_path)
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
        description="Run EasyOCR on a bank statement image and emit structured JSON.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the statement image file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to outputs/<stem>_ocr.json",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path("outputs") / f"{args.image.stem}_ocr.json"

    tokens = extract_tokens(args.image, gpu=not args.no_gpu)

    if not tokens:
        logger.warning("No tokens extracted — the image may be blank or unreadable.")
        sys.exit(0)

    save_ocr_json(tokens, args.output)
    print(f"Done. {len(tokens)} tokens -> {args.output}")
