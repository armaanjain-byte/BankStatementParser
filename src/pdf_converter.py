"""PDF rendering helpers for bank statement OCR."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
import numpy as np

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Render each page of a PDF statement to a grayscale image array."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    images: list[np.ndarray] = []

    with fitz.open(path) as document:
        logger.info("Rendering %d PDF page(s) from %s", document.page_count, path.name)
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
            )
            images.append(image.copy())

    return images
