"""
models/transaction_models.py — Transaction-layer data contracts.

Canonical location for Transaction, StatementMetadata, and BankStatement.
Re-exported from src.schema for backward compatibility.
"""

from __future__ import annotations

from src.schema import BankStatement, StatementMetadata, Transaction

__all__ = ["Transaction", "StatementMetadata", "BankStatement"]