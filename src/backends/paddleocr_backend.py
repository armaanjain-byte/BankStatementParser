"""
backends/paddleocr_backend.py — PaddleOCR backend for the Bank Statement Parser.

Wraps PaddleOCR and converts its output into the canonical List[OCRToken]
contract defined by OCRBackend.

PaddleOCR returns results in two formats depending on the installed version:
  • Dict-based (newer API):
        {'rec_texts': [...], 'rec_scores': [...], 'rec_boxes': [...]}
  • List-based (classic API):
        [[[bbox_points], (text, confidence)], ...]

Both are handled by _paddle_result_to_tokens().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

from src.backends.base import MIN_CONFIDENCE, OCRBackend
from src.schema import OCRToken

logger = logging.getLogger(__name__)

_ocr = None


def _get_ocr(use_gpu: bool):
    """Return a module-level PaddleOCR singleton, creating it on first use."""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR (gpu=%s)", use_gpu)
        device = "gpu" if use_gpu else "cpu"
        _ocr = PaddleOCR(lang="en", device=device, enable_mkldnn=False)
    return _ocr


def _paddle_result_to_tokens(result, min_confidence: float) -> List[OCRToken]:
    """Convert PaddleOCR result format to a canonical OCRToken list."""
    tokens: List[OCRToken] = []
    if result is None:
        return tokens

    for page in result:
        if page is None:
            continue

        if isinstance(page, dict):
            # Newer dict-based API
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", [])
            for i in range(len(rec_texts)):
                confidence = float(rec_scores[i])
                if confidence < min_confidence:
                    continue
                bbox = rec_boxes[i]
                tokens.append(
                    OCRToken(
                        text=rec_texts[i].strip(),
                        confidence=confidence,
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    )
                )
        else:
            # Classic list-based API: [[bbox_points], (text, confidence)]
            for line in page:
                bbox_points, (text, confidence) = line
                confidence = float(confidence)
                if confidence < min_confidence:
                    continue
                xs = [pt[0] for pt in bbox_points]
                ys = [pt[1] for pt in bbox_points]
                tokens.append(
                    OCRToken(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    )
                )

    logger.debug(
        "PaddleOCR extracted %d tokens (min_conf=%.2f)", len(tokens), min_confidence
    )
    return tokens


class PaddleOCRBackend(OCRBackend):
    """OCR backend that uses PaddleOCR for text recognition."""

    def extract_tokens(
        self,
        image_input: Union[str, Path, np.ndarray],
        min_confidence: float = MIN_CONFIDENCE,
        preprocess: bool = False,
    ) -> List[OCRToken]:
        """Run PaddleOCR on *image_input* and return canonical OCRToken list."""
        ocr = _get_ocr(use_gpu=self.gpu)

        if isinstance(image_input, np.ndarray):
            img_input: Union[str, np.ndarray] = image_input
        else:
            path = Path(image_input)
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

            img_input = str(path)
            if preprocess:
                import tempfile

                import cv2

                from src.preprocessor import preprocess_for_ocr

                arr = preprocess_for_ocr(path)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    cv2.imwrite(f.name, arr)
                    img_input = f.name

        result = ocr.ocr(img_input)
        return _paddle_result_to_tokens(result, min_confidence=min_confidence)