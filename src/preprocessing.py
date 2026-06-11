"""
preprocessor.py — OpenCV image preprocessing for OCR quality improvement.

Applies a sequence of image enhancements before the image is passed to the OCR backend.
Each step is optional and order-sensitive. Conservative defaults are used.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path


def preprocess_for_ocr(image_path: str | Path, debug: bool = False) -> np.ndarray:
    """
    Load and preprocess a bank statement image for OCR.

    Pipeline:
      1. Load in grayscale
      2. Deskew (correct scan rotation)
      3. Denoise
      4. Adaptive thresholding (binarize)
      5. Morphological cleanup (remove fine noise)
    
    Returns a uint8 numpy array ready for OCR.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img = _deskew(img)
    img = _denoise(img)
    img = _binarize(img)
    img = _morphological_cleanup(img)

    if debug:
        cv2.imwrite(str(image_path).replace(".", "_preprocessed."), img)

    return img
def detect_table_region(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Attempt to find the bounding box of the transaction table.
    Returns (x, y, w, h) or None if detection fails.
    
    Strategy: Find the largest contiguous rectangular region with
    dense horizontal lines, which is characteristic of ruled tables.
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=img.shape[1]//3, maxLineGap=20)
    if lines is None or len(lines) < 3:
        return None

    ys = sorted(set(int(line[0][1]) for line in lines))
    if len(ys) < 3:
        return None

    # Find the densest band of horizontal lines
    y_start = ys[0]
    y_end = ys[-1]
    return (0, y_start, img.shape[1], y_end - y_start)

def _deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct small rotation angles (±15 degrees)."""
    coords = np.column_stack(np.where(img < 128))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise(img: np.ndarray) -> np.ndarray:
    """Remove Gaussian and salt-and-pepper noise."""
    return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)


def _binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive thresholding handles uneven illumination."""
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def _morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """Remove isolated noise pixels with morphological opening."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)