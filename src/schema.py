"""
schema.py — Backward-compatibility layer for the Bank Statement Parser.

Rule 7: This module must remain as a backward-compatibility shim so that
existing imports such as:

    from src.schema import OCRToken
    from src.schema import Transaction
    from src.schema import BankStatement

continue to work without modification after the migration to the
models/ package.

All canonical definitions now live in:
    src/models/ocr_models.py        → OCRToken
    src/models/transaction_models.py → Transaction, StatementMetadata,
                                       BankStatement
    src/models/validation_models.py  → ValidationIssue, ValidationReport

This file re-exports every public name so that existing call sites are
unaffected.
"""

from __future__ import annotations

# ── OCR layer ─────────────────────────────────
from src.models.ocr_models import OCRToken

# ── Transaction / metadata layer ──────────────
from src.models.transaction_models import (
    BankStatement,
    StatementMetadata,
    Transaction,
)

# ── Validation layer ───────────────────────────
from src.models.validation_models import ValidationIssue, ValidationReport

__all__ = [
    "BankStatement",
    "OCRToken",
    "StatementMetadata",
    "Transaction",
    "ValidationIssue",
    "ValidationReport",
]
