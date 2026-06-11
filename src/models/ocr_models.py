"""
models/ocr_models.py — OCR-layer data structures.

Defines the canonical OCRToken that every backend must produce.
Downstream pipeline code imports OCRToken from here (or via the
schema.py backward-compatibility layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class OCRToken:
    """A single token produced by an OCR engine.

    Attributes:
        text:       The recognized text content of this token.
        confidence: OCR confidence score in the range [0.0, 1.0].
        bbox:       Bounding box as (x_min, y_min, x_max, y_max) in pixels.
                    Coordinates follow the top-left origin convention.
    """

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if len(self.bbox) != 4:
            raise ValueError(
                f"bbox must have exactly 4 elements, got {len(self.bbox)}"
            )