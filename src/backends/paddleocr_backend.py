"""
backends/paddleocr_backend.py — PaddleOCR implementation of BaseOCRBackend.

Wraps the existing PaddleOCR logic from ocr_paddle.py behind the common
backend interface.  All result-format handling (new dict API vs. legacy
list API) lives here so that downstream code only ever sees OCRToken.
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
#  Module-level OCR singleton (lazy-initialized)
# ──────────────────────────────────────────────

_ocr = None


def _get_ocr(use_gpu: bool = True):
    """Return the module-level PaddleOCR instance, creating it on first use.

    Args:
        use_gpu: Whether to request GPU acceleration.

    Returns:
        A ready-to-use ``PaddleOCR`` instance.
    """
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        device = "gpu" if use_gpu else "cpu"
        logger.info("Initializing PaddleOCR (gpu=%s)", use_gpu)
        _ocr = PaddleOCR(lang="en", device=device, enable_mkldnn=False, use_angle_cls=True)
    return _ocr


def _paddle_result_to_tokens(result, min_confidence: float) -> List[OCRToken]:
    """Convert a raw PaddleOCR result to a list of :class:`OCRToken`.

    Handles both the newer dict-based API and the legacy list-based API so
    that the backend remains compatible with different PaddleOCR versions.

    Args:
        result:         Raw return value of ``PaddleOCR.ocr()``.
        min_confidence: Drop tokens below this threshold.

    Returns:
        List of :class:`OCRToken` objects.
    """
    tokens: List[OCRToken] = []
    if result is None:
        return tokens

    for page in result:
        if page is None:
            continue

        # New dict-based API (PaddleOCR >= 2.8)
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", [])
            for i in range(len(rec_texts)):
                confidence = rec_scores[i]
                if confidence < min_confidence:
                    continue
                bbox = rec_boxes[i]
                tokens.append(
                    OCRToken(
                        text=rec_texts[i].strip(),
                        confidence=float(confidence),
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    )
                )
        else:
            # Legacy list-based API: [[bbox_points], (text, confidence)]
            for line in page:
                bbox_points, (text, confidence) = line
                if confidence < min_confidence:
                    continue
                xs = [pt[0] for pt in bbox_points]
                ys = [pt[1] for pt in bbox_points]
                tokens.append(
                    OCRToken(
                        text=text.strip(),
                        confidence=float(confidence),
                        bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    )
                )

    logger.debug(
        "PaddleOCR extracted %d tokens (min_conf=%.2f)", len(tokens), min_confidence
    )
    return tokens


class PaddleOCRBackend(BaseOCRBackend):
    """OCR backend that delegates to PaddleOCR."""

    def extract_tokens(
        self,
        image_path: Union[str, Path, np.ndarray],
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        preprocess: bool = False,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> List[OCRToken]:
        """Run PaddleOCR on *image_path* and return structured tokens.

        Args:
            image_path:     Path to the input image or a NumPy array.
            languages:      Ignored — PaddleOCR language is fixed to ``"en"``
                            at construction time.  Kept for API symmetry.
            gpu:            Whether to use GPU acceleration.
            preprocess:     Whether to run OpenCV preprocessing before OCR.
            min_confidence: Drop OCR tokens below this confidence score.

        Returns:
            A list of :class:`OCRToken` objects.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        ocr = _get_ocr(use_gpu=gpu)

        if isinstance(image_path, np.ndarray):
            img_input = image_path
        else:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            if not preprocess and self._is_low_quality(path):
                import cv2

                variance = float(np.var(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)))
                logger.warning(
                    "Low-quality image detected (variance=%s), enabling preprocessing automatically.",
                    f"{variance:.0f}",
                )
                preprocess = True

            img_input = str(path)
            if preprocess:
                import cv2
                import tempfile

                from src.preprocessing import preprocess_for_ocr

                arr = preprocess_for_ocr(path)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    cv2.imwrite(f.name, arr)
                    img_input = f.name

        result = ocr.ocr(img_input)
        return _paddle_result_to_tokens(result, min_confidence=min_confidence)