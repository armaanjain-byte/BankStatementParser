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

import re


# ------------------------------------------------------------------------------
#  Shared Extraction Patterns
# ------------------------------------------------------------------------------

_MONTH_FULL = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
    r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember|t)?"
    r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

_MONTH_SHORT = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

DATE_RE = re.compile(
    # Pattern 1: DD/MM/YYYY or DD-MM-YYYY
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 2: DD-Mon-YYYY or DD Mon YYYY
    r"\b\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 3: DD Mon or DD-Mon (missing year)
    rf"\b\d{{1,2}}[-\s]*(?:{_MONTH_FULL})(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 4: DDMonYYYY or MonYYYY (no spaces)
    rf"\b\d{{0,2}}(?:{_MONTH_SHORT})\d{{2,4}}(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 5: mixed separator DD.MM-YYYY or DD.MM.YYYY
    r"\b\d{1,2}\.\d{1,2}[-/]\d{2,4}(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 6: YYYY-MM-DD
    r"\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 7: Mon DD YYYY
    rf"\b(?:{_MONTH_FULL})\s+\d{{1,2}}\s+(?:\d{{2,4}})?(?:\b|(?=[A-Za-z0-9/_\-]))"
    r"|"
    # Pattern 8: DDMon (alphanumeric, no spaces, no year) e.g., 01Aug
    rf"\b\d{{1,2}}(?:{_MONTH_SHORT})(?:\b|(?=[A-Za-z0-9/_\-]))",
    re.IGNORECASE,
)

MONEY_RE = re.compile(r"\b\d+(?:,\d{3})*\.\d{2}\b")


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
_LEGACY_ALIAS_TO_COLUMN: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}

# Merged single source of truth for column aliases. The original constants.py
# aliases and the former layout_detector.py local aliases are combined here;
# overlapping aliases keep the higher-confidence value from either source.
_ALIAS_CONFIDENCES: dict[str, dict[str, float]] = {
    "date": {
        "date": 1.0,
        "tran date": 0.95,
        "txn date": 0.95,
        "transaction date": 0.95,
        "value date": 0.95,
        "posting date": 0.95,
    },
    "description": {
        "description": 1.0,
        "details": 1.0,
        "detuils": 0.70,
        "particulars": 1.0,
        "farticulars": 0.72,
        "remarks": 1.0,
        "narration": 1.0,
        "transaction details": 0.95,
        "transaction particulars": 0.95,
        "xriplion": 0.55,
    },
    "debit": {
        "debit": 1.0,
        "dehit": 0.70,
        "debit amount": 0.95,
        "withdrawal": 1.0,
        "withdrawals": 1.0,
        "dr": 1.0,
        "dr.": 1.0,
        "debit (dr.)": 0.95,
        "amount debited": 0.95,
    },
    "credit": {
        "credit": 1.0,
        "credit amount": 0.95,
        "deposit": 1.0,
        "deposits": 1.0,
        "cr": 1.0,
        "cr.": 1.0,
        "credit (cr.)": 0.95,
        "amount credited": 0.95,
    },
    "balance": {
        "balance": 1.0,
        "closing balance": 0.95,
        "running balance": 0.95,
        "available balance": 0.95,
        "bal": 1.0,
        "bal.": 1.0,
        "balnnct": 0.62,
        "kalance": 0.70,
        "pnnc": 0.35,
    },
    "amount": {
        "amount": 1.0,
        "amt": 0.95,
    },
    "instrument": {
        "instrument": 1.0,
        "instrumente": 0.75,
        "chq no": 0.90,
        "chy no": 0.65,
    },
    "type": {
        "type": 1.0,
    },
}

COLUMN_ALIASES = {
    canonical: frozenset(alias.title() for alias in aliases)
    for canonical, aliases in _ALIAS_CONFIDENCES.items()
}

ALIAS_TO_COLUMN: dict[str, tuple[str, float]] = {
    alias: (canonical, confidence)
    for canonical, aliases in _ALIAS_CONFIDENCES.items()
    for alias, confidence in aliases.items()
}


# ──────────────────────────────────────────────
#  Confidence Thresholds
# ──────────────────────────────────────────────
# Minimum OCR confidence scores used during extraction filtering.
# Tokens below these thresholds are flagged or discarded.

MIN_OCR_CONFIDENCE: float = 0.40          # Drop tokens below this
HIGH_CONFIDENCE_THRESHOLD: float = 0.85   # Consider "reliable" above this
