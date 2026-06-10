"""
backends/base.py — Abstract base class for OCR backends.

All OCR engines must subclass OCRBackend and implement extract_tokens().
The contract guarantees that downstream pipeline code receives identical
List[OCRToken] output regardless of which engine performed the recognition.

Adding a third engine requires only:
  1. Creating a new file backends/<engine>_backend.py
  2. Subclassing OCRBackend
  3. Implementing extract_tokens()

No changes to main.py, layout_detector, or any other pipeline stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np

from src.schema import OCRToken

MIN_CONFIDENCE: float = 0.10
LOW_QUALITY_VARIANCE_THRESHOLD: float = 1800.0


class OCRBackend(ABC):
    """Abstract OCR backend.  Subclasses must implement extract_tokens()."""

    def __init__(self, gpu: bool = True) -> None:
        self.gpu = gpu

    @abstractmethod
    def extract_tokens(
        self,
        image_input: Union[str, Path, np.ndarray],
        min_confidence: float = MIN_CONFIDENCE,
        preprocess: bool = False,
    ) -> List[OCRToken]:
        """Run OCR on *image_input* and return a list of OCRToken objects.

        Args:
            image_input:    Path to an image file, or a numpy array (uint8
                            grayscale or BGR) representing the image data.
            min_confidence: Drop tokens whose confidence falls below this
                            threshold.  Must be in [0.0, 1.0].
            preprocess:     If True, apply OpenCV preprocessing before
                            running OCR.  Backends may also auto-enable
                            preprocessing when low image quality is detected.

        Returns:
            A list of :class:`~src.schema.OCRToken` objects, one per
            detected text region, roughly in reading order.

        Raises:
            FileNotFoundError: If *image_input* is a path that does not exist.
        """

    # ── Shared helpers available to all backends ──────────────────────

    @staticmethod
    def is_low_quality(image_path: Union[str, Path]) -> bool:
        """Return True if the image likely needs preprocessing.

        Heuristic: clean digital-PDF renders have high pixel variance
        (black text on white background, score typically 3000–8000).
        Low-quality phone photos or faded scans have compressed histograms
        and score below the threshold.

        Args:
            image_path: Path to the image file to evaluate.

        Returns:
            True if the image variance is below the low-quality threshold.
        """
        import cv2  # imported lazily to avoid hard dep at module load

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        return float(np.var(img)) < LOW_QUALITY_VARIANCE_THRESHOLD