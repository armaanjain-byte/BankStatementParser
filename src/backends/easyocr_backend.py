"""
backends/easyocr_backend.py — EasyOCR backend for the Bank Statement Parser.

Wraps EasyOCR and converts its polygon-based output into the canonical
List[OCRToken] contract defined by OCRBackend.  All geometry is normalised
to axis-aligned bounding boxes (x_min, y_min, x_max, y_max).

EasyOCR output format per detection:
    (bbox, text, confidence)
    where bbox is a list of four [x, y] corner points (polygon).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

from src.backends.base import MIN_CONFIDENCE, OCRBackend
from src.schema import OCRToken

logger = logging.getLogger(__name__)

_reader = None


def _get_reader(languages: list[str], gpu: bool):
    """Return a module-level EasyOCR Reader singleton, creating it on first use."""
    global _reader
    if (
        _reader is None
        or getattr(_reader, "_init_langs", None) != languages
        or getattr(_reader, "_init_gpu", None) != gpu
    ):
        import easyocr

        logger.info("Initializing EasyOCR reader (languages=%s, gpu=%s)", languages, gpu)
        _reader = easyocr.Reader(languages, gpu=gpu)
        _reader._init_langs = languages
        _reader._init_gpu = gpu
    return _reader


def _polygon_to_bbox(polygon: list) -> tuple[int, int, int, int]:
    """Collapse an EasyOCR 4-corner polygon to an axis-aligned bounding box."""
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


class EasyOCRBackend(OCRBackend):
    """OCR backend that uses EasyOCR for text recognition."""

    def __init__(self, gpu: bool = True, languages: list[str] | None = None) -> None:
        super().__init__(gpu=gpu)
        self.languages = list(languages or ["en"])

    def extract_tokens(
        self,
        image_input: Union[str, Path, np.ndarray],
        min_confidence: float = MIN_CONFIDENCE,
        preprocess: bool = False,
    ) -> List[OCRToken]:
        """Run EasyOCR on *image_input* and return canonical OCRToken list."""
        image_name = "<array>"
        ocr_input: Union[str, np.ndarray]

        if isinstance(image_input, np.ndarray):
            ocr_input = image_input
        else:
            path = Path(image_input)
            image_name = path.name
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            if not preprocess and self.is_low_quality(path):
                import cv2

                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                variance = float(np.var(img)) if img is not None else 0.0
                logger.warning(
                    "Low-quality image detected (variance=%s), enabling preprocessing automatically.",
                    f"{variance:.0f}",
                )
                preprocess = True

            if preprocess:
                from src.preprocessor import preprocess_for_ocr

                ocr_input = preprocess_for_ocr(path)
            else:
                ocr_input = str(path)

        reader = _get_reader(self.languages, self.gpu)
        logger.info("Running EasyOCR on %s", image_name)
        raw_results = reader.readtext(ocr_input, detail=1, paragraph=False)

        tokens: List[OCRToken] = []
        for bbox_polygon, text, confidence in raw_results:
            bbox = _polygon_to_bbox(bbox_polygon)
            tokens.append(
                OCRToken(
                    text=text.strip(),
                    confidence=round(float(confidence), 4),
                    bbox=bbox,
                )
            )

        original_count = len(tokens)
        tokens = [t for t in tokens if t.confidence >= min_confidence]
        logger.debug(
            "Filtered %d low-confidence tokens (threshold=%s)",
            original_count - len(tokens),
            min_confidence,
        )
        logger.info("Extracted %d tokens from %s", len(tokens), image_name)
        return tokens