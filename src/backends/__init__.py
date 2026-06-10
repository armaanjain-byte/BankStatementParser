"""
backends/__init__.py — OCR backend factory.

Usage:
    from src.backends import get_backend
    backend = get_backend(engine="paddle", gpu=True)
    tokens = backend.extract_tokens(image_path)
"""

from __future__ import annotations

from src.backends.base import OCRBackend


def get_backend(engine: str, gpu: bool = True) -> OCRBackend:
    """Return an initialised OCR backend for the requested engine.

    Args:
        engine: One of ``"paddle"`` or ``"easyocr"``.
        gpu:    Whether to enable GPU acceleration.

    Returns:
        A concrete :class:`OCRBackend` instance ready to call
        :meth:`~OCRBackend.extract_tokens`.

    Raises:
        ValueError: If *engine* is not a recognised backend name.
    """
    if engine == "paddle":
        from src.backends.paddleocr_backend import PaddleOCRBackend

        return PaddleOCRBackend(gpu=gpu)
    if engine == "easyocr":
        from src.backends.easyocr_backend import EasyOCRBackend

        return EasyOCRBackend(gpu=gpu)
    raise ValueError(
        f"Unknown OCR engine: {engine!r}. Supported engines: 'paddle', 'easyocr'."
    )