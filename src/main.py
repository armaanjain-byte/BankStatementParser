"""
main.py — Batch OCR runner for the Bank Statement Parser.

Scans every image in ``test_set/``, runs EasyOCR via :mod:`src.ocr`,
and writes structured JSON output to ``outputs/``.

Usage:
    python -m src.main              # process all images
    python -m src.main --no-gpu     # CPU-only mode

This module does NOT perform:
  • Field extraction
  • Transaction reconstruction
  • Metadata extraction
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src.ocr import extract_tokens, save_ocr_json

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

INPUT_DIR = Path("test_set")
OUTPUT_DIR = Path("outputs")

# Image extensions to scan (case-insensitive via .suffix.lower())
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# ──────────────────────────────────────────────
#  Logging setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _average_confidence(tokens) -> float:
    """Return the mean confidence of a token list, or 0.0 if empty."""
    if not tokens:
        return 0.0
    return sum(t.confidence for t in tokens) / len(tokens)


def _output_path_for(image_path: Path) -> Path:
    """Map an image filename to its OCR JSON output path.

    Examples:
        23.jpg  -> outputs/23_ocr.json
        24.png  -> outputs/24_ocr.json
    """
    return OUTPUT_DIR / f"{image_path.stem}_ocr.json"


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def run(*, gpu: bool = True) -> None:
    """Process every image in INPUT_DIR and write OCR JSON to OUTPUT_DIR."""

    if not INPUT_DIR.exists():
        logger.error("Input directory not found: %s", INPUT_DIR)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect and sort image files
    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        logger.warning("No images found in %s", INPUT_DIR)
        sys.exit(0)

    logger.info("Found %d image(s) in %s", len(images), INPUT_DIR)
    logger.info("-" * 60)

    total_tokens = 0
    batch_start = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        img_start = time.perf_counter()

        # ── OCR ──────────────────────────────
        tokens = extract_tokens(image_path, gpu=gpu)
        avg_conf = _average_confidence(tokens)
        total_tokens += len(tokens)

        # ── Save ─────────────────────────────
        out_path = _output_path_for(image_path)
        save_ocr_json(tokens, out_path)

        elapsed = time.perf_counter() - img_start

        # ── Log ──────────────────────────────
        logger.info(
            "[%d/%d] %-20s | tokens: %4d | avg conf: %.4f | %.2fs",
            idx,
            len(images),
            image_path.name,
            len(tokens),
            avg_conf,
            elapsed,
        )

    batch_elapsed = time.perf_counter() - batch_start

    logger.info("-" * 60)
    logger.info(
        "Done. %d image(s), %d total tokens, %.2fs elapsed.",
        len(images),
        total_tokens,
        batch_elapsed,
    )


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch OCR: test_set/ images -> outputs/ JSON.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )
    args = parser.parse_args()

    run(gpu=not args.no_gpu)