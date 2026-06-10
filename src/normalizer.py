"""
normalizer.py — Normalization orchestrator for the Bank Statement Parser.

Rule 3: This module is an orchestrator only.  No normalization logic lives
here; it delegates entirely to the sub-modules in src/normalizers/.

Backward-compatible public API (unchanged):
    normalize_date_strings(rows)   → (rows, int)
    normalize_amount_strings(rows) → (rows, int)

These two names are re-exported directly from the normalizers package so
that all existing call sites continue to work without modification.
"""

from __future__ import annotations

# Re-export the two public functions that the rest of the pipeline imports.
from src.normalizers.amount_normalizer import normalize_amount_strings
from src.normalizers.date_normalizer import normalize_date_strings

__all__ = [
    "normalize_amount_strings",
    "normalize_date_strings",
]
