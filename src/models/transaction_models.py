"""
models/transaction_models.py — Transaction-layer data structures.

Defines Transaction, StatementMetadata, and BankStatement.
These are the canonical models produced by the extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from src.models.validation_models import ValidationReport


@dataclass
class Transaction:
    """A single financial transaction extracted from a bank statement.

    At minimum a transaction should have a date, but during reconstruction
    some fields may be unavailable, so everything is Optional.

    Attributes:
        date:        Transaction date as it appears on the statement
                     (e.g. "01/04/2025", "01 Apr 2025").  Kept as a raw
                     string; date normalization is a downstream concern.
        description: Narration / particulars / remarks text.
        amount:      Generic transaction amount when direction is unknown.
        debit:       Amount debited (withdrawn), or None if not applicable.
        credit:      Amount credited (deposited), or None if not applicable.
        balance:     Running balance after this transaction, if available.
    """

    date: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    amount: Optional[float] = field(default=None)
    debit: Optional[float] = field(default=None)
    credit: Optional[float] = field(default=None)
    balance: Optional[float] = field(default=None)


@dataclass
class StatementMetadata:
    """Header-level metadata extracted from the bank statement.

    These fields are typically found in the letterhead or summary block
    at the top of a statement page.

    Attributes:
        bank_name:             Name of the issuing bank.
        account_holder:        Name of the account holder.
        account_number:        Full or masked account number.
        statement_period_from: Start date of the statement period (raw string).
        statement_period_to:   End date of the statement period (raw string).
        currency:              ISO-4217 currency code (e.g. "INR", "USD").
    """

    bank_name: Optional[str] = field(default=None)
    account_holder: Optional[str] = field(default=None)
    account_number: Optional[str] = field(default=None)
    statement_period_from: Optional[str] = field(default=None)
    statement_period_to: Optional[str] = field(default=None)
    currency: Optional[str] = field(default=None)


@dataclass
class BankStatement:
    """Complete parsed representation of a bank statement.

    This is the final output of the parsing pipeline and the shape of the
    standardized JSON that gets written to disk.

    Attributes:
        metadata:     Statement-level metadata (bank name, account info, etc.).
        transactions: Ordered list of transactions extracted from the statement.
        validation:   The validation report for the extraction process.
    """

    metadata: StatementMetadata = field(default_factory=StatementMetadata)
    transactions: List[Transaction] = field(default_factory=list)
    validation: ValidationReport = field(default_factory=ValidationReport)

    # ── Convenience helpers ──────────────────

    def to_dict(self) -> dict:
        """Serialize the entire statement to a plain dict (JSON-ready)."""
        return asdict(self)

    @property
    def transaction_count(self) -> int:
        """Number of transactions currently held."""
        return len(self.transactions)