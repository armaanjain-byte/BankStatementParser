"""
backends/base.py — Abstract base class that every OCR backend must implement.

Rule 1: All OCR engines must return the same OCRToken contract.
Rule 2: Downstream pipeline code must not know which engine produced the tokens.

Any new backend (Tesseract, Google Vision, etc.) must subclass BaseOCRBackend
and implement extract_tokens() to remain pipeline-compatible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.models.ocr_models import OCRToken


class BaseOCRBackend(ABC):
    """Abstract OCR backend.

    Subclasses must implement :meth:`extract_tokens`.  All other methods
    have default implementations that may be overridden where needed.
    """

    # ──────────────────────────────────────────────
    #  Abstract interface
    # ──────────────────────────────────────────────

    @abstractmethod
    def extract_tokens(
        self,
        image_path: Union[str, Path, np.ndarray],
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        preprocess: bool = False,
        min_confidence: float = 0.10,
    ) -> List[OCRToken]:
        """Run OCR on *image_path* and return a list of :class:`OCRToken`.

        Args:
            image_path:     Path to an image file, or a NumPy array already
                            loaded into memory (HxWxC, uint8, BGR).
            languages:      Language code list understood by the underlying
                            engine.  Defaults to ``["en"]`` when *None*.
            gpu:            Whether to request GPU acceleration.
            preprocess:     Whether to run OpenCV preprocessing before OCR.
            min_confidence: Drop tokens whose confidence is below this value.

        Returns:
            Ordered list of :class:`OCRToken` objects (roughly top-to-bottom,
            left-to-right, matching the reading order of the page).

        Raises:
            FileNotFoundError: If *image_path* is a path that does not exist.
        """

    # ──────────────────────────────────────────────
    #  Shared helper: low-quality image detection
    # ──────────────────────────────────────────────

    @staticmethod
    def _is_low_quality(image_path: Union[str, Path]) -> bool:
        """Return *True* if the image likely needs preprocessing.

        Heuristic: digital bank statement scans have high pixel variance
        (sharp black text on white background).  Phone photos and faded
        scans have compressed, noisy histograms with lower variance.

        Args:
            image_path: Path to the image file.

        Returns:
            *True* when the image variance is below the tuned threshold.
        """
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        variance = float(np.var(img))
        # Digital statements typically score 3 000 – 8 000; phone photos < 1 800.
        return variance < 1800

    # ──────────────────────────────────────────────
    #  Shared helper: polygon → axis-aligned bbox
    # ──────────────────────────────────────────────

    @staticmethod
    def _polygon_to_bbox(
        polygon: List[List[Union[int, float]]],
    ) -> tuple[int, int, int, int]:
        """Convert a four-corner polygon to an axis-aligned bounding box.

        Args:
            polygon: List of four ``[x, y]`` coordinate pairs (any winding
                     order).

        Returns:
            ``(x_min, y_min, x_max, y_max)`` as integers.
        """
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))