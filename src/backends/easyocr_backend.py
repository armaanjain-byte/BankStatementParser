"""
backends/easyocr_backend.py — EasyOCR implementation of BaseOCRBackend.

Wraps the existing EasyOCR logic from ocr.py behind the common backend
interface so that downstream code only depends on OCRToken, never on
EasyOCR internals.

Public API mirrors the original ocr.extract_tokens() signature so that
callers can switch backends without changing their argument lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.backends.base import BaseOCRBackend
from src.models.ocr_models import OCRToken

logger = logging.getLogger(__name__)

MIN_CONFIDENCE: float = 0.10

# ──────────────────────────────────────────────
#  Module-level reader singleton (lazy-initialized)
# ──────────────────────────────────────────────
# EasyOCR model loading is expensive (~2-5 s on first call).
# We keep a module-level singleton so repeated calls within the
# same process don't pay the cost more than once.

_reader = None


def _get_reader(languages: List[str], gpu: bool):
    """Return the module-level EasyOCR Reader, creating it on first use.

    Args:
        languages: Language codes to load.
        gpu:       Whether to attempt GPU acceleration.

    Returns:
        A ready-to-use ``easyocr.Reader`` instance.
    """
    global _reader
    init_langs = getattr(_reader, "_init_langs", None)
    init_gpu = getattr(_reader, "_init_gpu", None)

    if _reader is None or init_langs != languages or init_gpu != gpu:
        import easyocr

        logger.info("Initializing EasyOCR reader (languages=%s, gpu=%s)", languages, gpu)
        _reader = easyocr.Reader(languages, gpu=gpu)
        _reader._init_langs = languages
        _reader._init_gpu = gpu
    return _reader


class EasyOCRBackend(BaseOCRBackend):
    """OCR backend that delegates to EasyOCR."""

    def extract_tokens(
        self,
        image_path: Union[str, Path, np.ndarray],
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        preprocess: bool = False,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> List[OCRToken]:
        """Run EasyOCR on *image_path* and return structured tokens.

        Args:
            image_path:     Path to the input image or a NumPy array.
            languages:      Language codes for EasyOCR (default: ``["en"]``).
            gpu:            Whether to use GPU acceleration.
            preprocess:     Whether to run OpenCV preprocessing before OCR.
            min_confidence: Drop OCR tokens below this confidence score.

        Returns:
            A list of :class:`OCRToken` objects ordered by EasyOCR's default
            reading-order heuristic (roughly top-to-bottom, left-to-right).

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        langs = list(languages or ["en"])
        image_name = "<array>"
        ocr_input: Union[str, np.ndarray]

        if isinstance(image_path, np.ndarray):
            ocr_input = image_path
        else:
            path = Path(image_path)
            image_name = path.name
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            if not preprocess and self._is_low_quality(path):
                variance = float(np.var(__import__("cv2").imread(str(path), 0)))
                logger.warning(
                    "Low-quality image detected (variance=%s), enabling preprocessing automatically.",
                    f"{variance:.0f}",
                )
                preprocess = True

            ocr_input = (
                __import__("archive.preprocessor", fromlist=["preprocess_for_ocr"]).preprocess_for_ocr(path)
                if preprocess
                else str(path)
            )

        reader = _get_reader(langs, gpu)
        logger.info("Running EasyOCR on %s", image_name)
        raw_results = reader.readtext(ocr_input, detail=1, paragraph=False)

        tokens: List[OCRToken] = []
        for bbox_polygon, text, confidence in raw_results:
            bbox = self._polygon_to_bbox(bbox_polygon)
            tokens.append(
                OCRToken(
                    text=text.strip(),
                    confidence=round(float(confidence), 4),
                    bbox=bbox,
                )
            )

        original_count = len(tokens)
        tokens = [t for t in tokens if t.confidence >= min_confidence]
        filtered_count = original_count - len(tokens)
        logger.debug(
            "Filtered %d low-confidence tokens (threshold=%s)",
            filtered_count,
            min_confidence,
        )

        logger.info("Extracted %d tokens from %s", len(tokens), image_name)
        return tokens