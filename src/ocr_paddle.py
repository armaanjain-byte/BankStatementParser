"""
ocr_paddle.py — PaddleOCR backend for the Bank Statement Parser.
Drop-in replacement for ocr.py with identical public API.
PaddleOCR provides better spatial accuracy for tabular financial documents.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Union
import numpy as np
from src.schema import OCRToken

logger = logging.getLogger(__name__)

_ocr = None

def _get_ocr(use_gpu: bool = True):
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR (gpu=%s)", use_gpu)
        device = "gpu" if use_gpu else "cpu"
        _ocr = PaddleOCR(lang="en", device=device, enable_mkldnn=False)
    return _ocr

def _paddle_result_to_tokens(result, min_confidence: float = 0.10) -> List[OCRToken]:
    """Convert PaddleOCR result format to OCRToken list."""
    tokens = []
    if result is None:
        return tokens
    for page in result:
        if page is None:
            continue
        if hasattr(page, 'keys') or isinstance(page, dict):
            rec_texts = page.get('rec_texts', [])
            rec_scores = page.get('rec_scores', [])
            rec_boxes = page.get('rec_boxes', [])
            for i in range(len(rec_texts)):
                text = rec_texts[i]
                confidence = rec_scores[i]
                bbox = rec_boxes[i]
                if confidence < min_confidence:
                    continue
                bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                tokens.append(OCRToken(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=bbox_tuple,
                ))
        else:
            for line in page:
                # PaddleOCR format: [[bbox_points], (text, confidence)]
                bbox_points, (text, confidence) = line
                if confidence < min_confidence:
                    continue
                xs = [pt[0] for pt in bbox_points]
                ys = [pt[1] for pt in bbox_points]
                tokens.append(OCRToken(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                ))
    logger.debug("PaddleOCR extracted %d tokens (min_conf=%.2f)", len(tokens), min_confidence)
    return tokens

def _is_low_quality(image_path) -> bool:
    """Return True if the image likely needs preprocessing (phone photo, faded scan)."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    variance = float(np.var(img))
    return variance < 1800  # tuned threshold; digital statements typically score 3000-8000


def extract_tokens(
    image_path: Union[str, Path, np.ndarray],
    languages: list[str] | None = None,
    gpu: bool = True,
    min_confidence: float = 0.10,
    preprocess: bool = False,
) -> List[OCRToken]:
    ocr = _get_ocr(use_gpu=gpu)
    
    if isinstance(image_path, np.ndarray):
        img_input = image_path
    else:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        if not preprocess:
            if _is_low_quality(path):
                import cv2
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                variance = float(np.var(img)) if img is not None else 0.0
                logger.warning(
                    "Low-quality image detected (variance=%s), enabling preprocessing automatically.",
                    f"{variance:.0f}"
                )
                preprocess = True
                
        img_input = str(path)
        if preprocess:
            from archive.preprocessor import preprocess_for_ocr
            import tempfile, cv2
            arr = preprocess_for_ocr(path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, arr)
                img_input = f.name
                
    result = ocr.ocr(img_input)
    return _paddle_result_to_tokens(result, min_confidence=min_confidence)
