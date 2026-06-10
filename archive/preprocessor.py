"""OpenCV preprocessing helpers for OCR input images."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_for_ocr(image_path: str | Path, debug: bool = False) -> np.ndarray:
    """Load and preprocess a bank statement image for OCR."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    image = _deskew(image)
    image = cv2.fastNlMeansDenoising(
        image,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    if debug:
        debug_path = path.with_name(f"{path.stem}_preprocessed{path.suffix}")
        cv2.imwrite(str(debug_path), image)
        logger.debug("Saved preprocessed image -> %s", debug_path)

    return image


def _deskew(image: np.ndarray) -> np.ndarray:
    dark_pixel_coords = np.column_stack(np.where(image < 128))
    if len(dark_pixel_coords) < 10:
        return image

    angle = cv2.minAreaRect(dark_pixel_coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    angle = max(-45.0, min(45.0, float(angle)))

    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
