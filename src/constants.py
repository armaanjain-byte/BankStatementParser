"""
constants.py — Shared constants for the Bank Statement Parser.

This module is the single source of truth for:
  • banks the parser currently supports,
  • currency code ↔ symbol mappings, and
  • the many aliases that different banks use for common column headers
    (e.g. "Withdrawal" vs "Debit" vs "Dr").

Naming convention
─────────────────
• Sets  → SCREAMING_SNAKE  (look-ups / membership tests)
• Dicts → SCREAMING_SNAKE  (mappings)
• All collections are **frozenset** or **tuple** where mutation is
  never expected, making accidental edits a loud error.

Extending
─────────
When a new bank or currency is encountered during dataset expansion,
add it here *before* touching extraction logic so the whole pipeline
stays consistent.
"""

from __future__ import annotations


# ──────────────────────────────────────────────
#  Supported Banks
# ──────────────────────────────────────────────
# Banks whose statement layouts have been analyzed and for which
# extraction heuristics exist (or are planned).

SUPPORTED_BANKS: frozenset[str] = frozenset({
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Punjab National Bank",
    "Bank of Baroda",
    "Canara Bank",
    "Union Bank of India",
    "Kotak Mahindra Bank",
    "IndusInd Bank",
})


# ──────────────────────────────────────────────
#  Currency Mappings
# ──────────────────────────────────────────────
# ISO-4217 code → common symbol / abbreviation.
# Used for display formatting and for detecting currency from OCR text.

CURRENCY_MAP: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AED": "د.إ",
    "SGD": "S$",
    "AUD": "A$",
    "CAD": "C$",
}

# Reverse look-up: symbol → ISO code  (built automatically)
SYMBOL_TO_CURRENCY: dict[str, str] = {
    symbol: code for code, symbol in CURRENCY_MAP.items()
}


# ──────────────────────────────────────────────
#  Transaction Column Aliases
# ──────────────────────────────────────────────
# Different banks label the same logical column with different headers.
# Each group below maps a *canonical* column name to all known aliases
# (case-insensitive matching is assumed at extraction time).
#
# The canonical names mirror the Transaction dataclass fields in schema.py.

COLUMN_ALIASES: dict[str, frozenset[str]] = {
    "date": frozenset({
        "Date",
        "Txn Date",
        "Transaction Date",
        "Value Date",
        "Posting Date",
    }),

    "description": frozenset({
        "Description",
        "Narration",
        "Particulars",
        "Details",
        "Remarks",
        "Transaction Details",
        "Transaction Particulars",
    }),

    "debit": frozenset({
        "Debit",
        "Debit Amount",
        "Withdrawal",
        "Withdrawals",
        "Dr",
        "Dr.",
        "Debit (Dr.)",
        "Amount Debited",
    }),

    "credit": frozenset({
        "Credit",
        "Credit Amount",
        "Deposit",
        "Deposits",
        "Cr",
        "Cr.",
        "Credit (Cr.)",
        "Amount Credited",
    }),

    "balance": frozenset({
        "Balance",
        "Closing Balance",
        "Running Balance",
        "Available Balance",
        "Bal",
        "Bal.",
    }),
}

# Flattened reverse look-up: lowered alias → canonical column name.
# Useful for quick header matching during table reconstruction.
ALIAS_TO_COLUMN: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


# ──────────────────────────────────────────────
#  Confidence Thresholds
# ──────────────────────────────────────────────
# Minimum OCR confidence scores used during extraction filtering.
# Tokens below these thresholds are flagged or discarded.

MIN_OCR_CONFIDENCE: float = 0.40          # Drop tokens below this
HIGH_CONFIDENCE_THRESHOLD: float = 0.85   # Consider "reliable" above this
