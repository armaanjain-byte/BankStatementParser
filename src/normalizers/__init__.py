"""
normalizers — OCR output normalization module.

Provides functions to clean up and normalize OCR-extracted text:
- Date string normalization (OCR corrections, interpolation, canonicalization)
- Amount string normalization (decimal handling, character substitution)
"""

from src.normalizers.date_normalizer import normalize_date_strings
from src.normalizers.amount_normalizer import normalize_amount_strings

__all__ = [
    "normalize_date_strings",
    "normalize_amount_strings",
]