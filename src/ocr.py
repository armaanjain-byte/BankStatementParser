"""
ocr.py — OCR layer for the Bank Statement Parser.

Responsibilities (and *only* these):
  1. Accept a path to a bank statement image.
  2. Run EasyOCR to extract every text detection.
  3. Convert raw EasyOCR results into a list of OCRToken objects.
  4. Optionally serialize those tokens to a human-readable JSON file.

This module does NOT perform:
  • Field extraction
  • Transaction parsing
  • Any form of validation

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


import easyocr
import numpy as np

from archive.preprocessor import preprocess_for_ocr
from src.schema import OCRToken

logger = logging.getLogger(__name__)
MIN_CONFIDENCE = 0.10

# ──────────────────────────────────────────────
#  Module-level reader (lazy-initialized)
# ──────────────────────────────────────────────
# EasyOCR model loading is expensive (~2-5 s on first call).
# We keep a module-level singleton so repeated calls within the
# same process don't pay the cost more than once.

_reader: easyocr.Reader | None = None


def _get_reader(languages: list[str] | None = None, gpu: bool = True) -> easyocr.Reader:
    """Return the module-level EasyOCR Reader, creating it on first use.

    Args:
        languages: Language codes to load (default: ``["en"]``).
        gpu:       Whether to attempt GPU acceleration.

    Returns:
        A ready-to-use ``easyocr.Reader`` instance.
    """
    global _reader
    langs = list(languages or ["en"])
    init_langs = getattr(_reader, "_init_langs", None)
    init_gpu = getattr(_reader, "_init_gpu", None)

    if _reader is None or init_langs != langs or init_gpu != gpu:
        logger.info("Initializing EasyOCR reader (languages=%s, gpu=%s)", langs, gpu)
        _reader = easyocr.Reader(langs, gpu=gpu)
        _reader._init_langs = langs
        _reader._init_gpu = gpu
    return _reader


# ──────────────────────────────────────────────
#  Polygon → Axis-aligned bbox
# ──────────────────────────────────────────────

def _polygon_to_bbox(polygon: list[list[int | float]]) -> tuple[int, int, int, int]:
    """Convert an EasyOCR polygon to an axis-aligned bounding box.

    EasyOCR returns each detection's location as four [x, y] corner
    points (top-left, top-right, bottom-right, bottom-left).  We
    collapse these into the tightest axis-aligned rectangle.

    Args:
        polygon: List of four ``[x, y]`` coordinate pairs.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` as integers.
    """
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


# ──────────────────────────────────────────────
#  Core public API
# ──────────────────────────────────────────────

def _is_low_quality(image_path) -> bool:
    """Return True if the image likely needs preprocessing (phone photo, faded scan)."""
    import cv2
    import numpy as np
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    # Bimodal check: clean digital PDFs have high pixel variance (black text on white bg).
    # Low-quality scans / photos have compressed, noisy histograms.
    variance = float(np.var(img))
    return variance < 1800  # tuned threshold; digital statements typically score 3000-8000


def extract_tokens(
    image_path: Union[str, Path, np.ndarray],
    languages: list[str] | None = None,
    gpu: bool = True,
    preprocess: bool = False,
    min_confidence: float = MIN_CONFIDENCE,
) -> List[OCRToken]:
    """Run OCR on a bank statement image and return structured tokens.

    Args:
        image_path: Path to the input image (JPEG, PNG, TIFF, etc.).
        languages:  Language codes for EasyOCR (default: ``["en"]``).
        gpu:        Whether to use GPU acceleration.
        preprocess: Whether to run OpenCV preprocessing before OCR.
        min_confidence: Drop OCR tokens below this confidence score.

    Returns:
        A list of :class:`OCRToken` objects, one per detected text region,
        ordered by EasyOCR's default reading-order heuristic (roughly
        top-to-bottom, left-to-right).

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    image_name = "<array>"
    ocr_input: str | np.ndarray

    if isinstance(image_path, np.ndarray):
        ocr_input = image_path
    else:
        path = Path(image_path)
        image_name = path.name
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        if not preprocess:
            if _is_low_quality(path):
                import cv2
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                variance = float(np.var(img)) if img is not None else 0.0
                logger.warning(
                    "Low-quality image detected (variance=%s), enabling preprocessing automatically.",
                    f"{variance:.0f}"
                )
                preprocess = True
                
        ocr_input = preprocess_for_ocr(path) if preprocess else str(path)

    reader = _get_reader(languages=languages, gpu=gpu)
    logger.info("Running OCR on %s", image_name)
    raw_results = reader.readtext(ocr_input, detail=1, paragraph=False)

    tokens: List[OCRToken] = []
    for bbox_polygon, text, confidence in raw_results:
        bbox = _polygon_to_bbox(bbox_polygon)
        token = OCRToken(
            text=text.strip(),
            confidence=round(float(confidence), 4),
            bbox=bbox,
        )
        tokens.append(token)

    original_count = len(tokens)
    tokens = [token for token in tokens if token.confidence >= min_confidence]
    filtered_count = original_count - len(tokens)
    logger.debug(
        "Filtered %d low-confidence tokens (threshold=%s)",
        filtered_count,
        min_confidence,
    )

    logger.info(
        "Extracted %d tokens from %s",
        len(tokens),
        image_name,
    )
    return tokens


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

    # Derive default output path if not specified
    if args.output is None:
        args.output = Path("outputs") / f"{args.image.stem}_ocr.json"

    tokens = extract_tokens(args.image, gpu=not args.no_gpu)

    if not tokens:
        logger.warning("No tokens extracted — the image may be blank or unreadable.")
        sys.exit(0)

    save_ocr_json(tokens, args.output)
    print(f"Done. {len(tokens)} tokens -> {args.output}")
