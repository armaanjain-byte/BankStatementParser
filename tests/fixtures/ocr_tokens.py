"""
tests/fixtures/ocr_tokens.py — Builders for OCRToken and Row test fixtures.

These helpers let unit tests construct minimal, deterministic pipeline
inputs without depending on a real OCR run (paddleocr is not installed
in this environment, and no *_ocr.json caches ship with the repo).
"""

from __future__ import annotations

from typing import List, Tuple

from src.schema import OCRToken
from src.transaction_reconstructor import Row, reconstruct_rows


def make_token(
    text: str,
    confidence: float = 0.9,
    bbox: Tuple[int, int, int, int] = (0, 0, 10, 10),
) -> OCRToken:
    """Build a single OCRToken with sensible defaults."""
    return OCRToken(text=text, confidence=confidence, bbox=bbox)


def make_row_from_text(text: str, y_center: float = 100.0) -> Row:
    """Build a Row directly from a text string (no real tokens).

    Useful for normalizer/parser/classifier tests that only inspect
    ``row.text`` / ``row.original_text`` and don't need token geometry.
    """
    return Row(y_center=y_center, tokens=[], text=text, original_text=text)


def make_row_from_words(
    words: List[str],
    y: int = 100,
    x_start: int = 0,
    word_width: int = 40,
    gap: int = 5,
    confidence: float = 0.9,
) -> Row:
    """Build a Row from a list of words, auto-generating left-to-right bboxes.

    Each word becomes one OCRToken at height 20px, spaced ``gap`` px apart.
    Useful for layout_detector / semantic_assigner tests that need real
    token geometry (x_center matching).
    """
    tokens = []
    x = x_start
    for word in words:
        bbox = (x, y, x + word_width, y + 20)
        tokens.append(make_token(word, confidence=confidence, bbox=bbox))
        x += word_width + gap

    rows = reconstruct_rows(tokens, threshold=15.0)
    assert len(rows) == 1, "make_row_from_words must produce exactly one row"
    return rows[0]


def make_tokens_in_rows(rows_of_words: List[List[str]], row_height: int = 30) -> List[OCRToken]:
    """Build a flat OCRToken list spanning multiple rows for reconstruction tests.

    Args:
        rows_of_words: list of rows, each a list of word strings.
        row_height: vertical spacing between rows (px).

    Returns:
        Flat list of OCRToken suitable for reconstruct_rows().
    """
    tokens: List[OCRToken] = []
    for row_idx, words in enumerate(rows_of_words):
        y = row_idx * row_height
        x = 0
        for word in words:
            tokens.append(make_token(word, confidence=0.9, bbox=(x, y, x + 40, y + 20)))
            x += 45
    return tokens