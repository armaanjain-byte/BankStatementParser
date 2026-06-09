"""
header_extractor.py — Deterministic metadata extraction from OCR tokens.

Extracts statement-level metadata from the header region of bank statements
using **rule-based heuristics only** — no LLMs, no external APIs.

Strategy
────────
Bank statements follow a common visual layout: the top ~30 % of the page
holds the bank logo, account holder details, account number, statement
period, and currency.  We exploit this by:

  1. Restricting most searches to the *header zone* (top portion of the
     page, estimated from bounding-box y-coordinates).
  2. Matching known bank names, label patterns, and date formats with
     fuzzy/substring matching to handle OCR noise.
  3. Assigning a per-field confidence score that reflects both OCR
     confidence and extraction-rule reliability.

This module does NOT parse transactions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.constants import (
    CURRENCY_MAP,
    SUPPORTED_BANKS,
    SYMBOL_TO_CURRENCY,
)
from src.schema import OCRToken, StatementMetadata

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Extraction result with per-field confidence
# ──────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Wrapper that pairs a :class:`StatementMetadata` with per-field
    confidence scores indicating how reliable each extraction was.

    Confidence values are in [0.0, 1.0]:
      • 1.0 = high-confidence exact match
      • 0.0 = not extracted (field is None)

    Attributes:
        metadata:    The extracted :class:`StatementMetadata`.
        confidence:  Dict mapping field names to confidence scores.
    """

    metadata: StatementMetadata = field(default_factory=StatementMetadata)
    confidence: dict[str, float] = field(default_factory=lambda: {
        "bank_name": 0.0,
        "account_holder": 0.0,
        "account_number": 0.0,
        "statement_period_from": 0.0,
        "statement_period_to": 0.0,
        "currency": 0.0,
    })

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "metadata": asdict(self.metadata),
            "confidence": self.confidence,
        }


# ──────────────────────────────────────────────
#  Bank-name detection knowledge base
# ──────────────────────────────────────────────
# Maps lowercased OCR-friendly substrings → canonical bank name.
# Includes common OCR mis-reads and abbreviations.

_BANK_KEYWORDS: dict[str, str] = {}

# Build from SUPPORTED_BANKS
for _bank in SUPPORTED_BANKS:
    _BANK_KEYWORDS[_bank.lower()] = _bank

# Short forms, abbreviations, and OCR-mangled variants
_BANK_KEYWORDS.update({
    "sbi":                "State Bank of India",
    "state bank":         "State Bank of India",
    "hdfc":               "HDFC Bank",
    "icici":              "ICICI Bank",
    "axis bank":          "Axis Bank",
    "axis":               "Axis Bank",
    "punjab national":    "Punjab National Bank",
    "punjabnational":     "Punjab National Bank",
    "punjabnationol":     "Punjab National Bank",    # OCR mis-read
    "pnb":                "Punjab National Bank",
    "bank of baroda":     "Bank of Baroda",
    "canara":             "Canara Bank",
    "union bank":         "Union Bank of India",
    "kotak":              "Kotak Mahindra Bank",
    "indusind":           "IndusInd Bank",
    # International banks commonly seen in dataset
    "hsbc":               "HSBC",
    "barclays":           "Barclays",
    "standard chartered": "Standard Chartered",
    "citibank":           "Citibank",
    "citi":               "Citibank",
    "wells fargo":        "Wells Fargo",
    "chase":              "JPMorgan Chase",
    "jpmorgan":           "JPMorgan Chase",
    "bank of america":    "Bank of America",
    "westpac":            "Westpac",
    "natwest":            "NatWest",
    "lloyds":             "Lloyds Bank",
    "deutsche bank":      "Deutsche Bank",
    "yes bank":           "Yes Bank",
    "idbi":               "IDBI Bank",
    "federal bank":       "Federal Bank",
    "bandhan":            "Bandhan Bank",
})

# Sort keywords longest-first so more specific matches win
_BANK_KEYWORDS_SORTED: list[tuple[str, str]] = sorted(
    _BANK_KEYWORDS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# ──────────────────────────────────────────────
#  Currency detection knowledge base
# ──────────────────────────────────────────────

_CURRENCY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Explicit label: "Currency INR", "Currency: USD"
    (re.compile(r"currency\s*:?\s*([A-Z]{3})", re.IGNORECASE), None),  # group(1) = code
    # Symbol detection
    (re.compile(r"₹"),              "INR"),
    (re.compile(r"\bRs\.?\b"),      "INR"),
    (re.compile(r"\bINR\b"),        "INR"),
    (re.compile(r"\$"),             "USD"),
    (re.compile(r"€"),              "EUR"),
    (re.compile(r"£"),              "GBP"),
    (re.compile(r"¥"),              "JPY"),
]

# ──────────────────────────────────────────────
#  Account number patterns
# ──────────────────────────────────────────────

_ACCOUNT_LABEL_RE = re.compile(
    r"(?:account\s*(?:no|number|#)?|a/c\s*(?:no|number|#)?|acct)\s*"
    r":?\s*"
    r"([0-9X*]+(?:[0-9X*\s\-]+[0-9X*])?)",
    re.IGNORECASE,
)

_STATEMENT_ACCOUNT_RE = re.compile(
    r"statement\s+of\s+account\s*:?\s*"
    r"([0-9X*]+(?:[0-9X*\s\-]+[0-9X*])?)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
#  Account holder patterns
# ──────────────────────────────────────────────

_HOLDER_LABEL_RE = re.compile(
    r"(?:customer\s*name|account\s*holder|name\s*of\s*account\s*holder"
    r"|a/c\s*holder|mr\.?|mrs\.?|ms\.?|m/s)"
    r"\s*:?\s*"
    r"(.+)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
#  Statement period patterns
# ──────────────────────────────────────────────

# "For Period: 26/9/2019 to 25/12/2019"
_PERIOD_EXPLICIT_RE = re.compile(
    r"(?:for\s+(?:the\s+)?period|period)\s*:?\s*"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+"
    r"(?:to|through|-)\s+"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    re.IGNORECASE,
)

# "(From DD-MM-YYYY  DD-MM-YYYY)"  or  "From DD-MM-YYYY to DD-MM-YYYY"
_PERIOD_FROM_TO_RE = re.compile(
    r"\(?from\s+"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+"
    r"(?:to\s+)?"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\)?",
    re.IGNORECASE,
)

# "Statement Date: 30th June 2015" / "Statement Period: Jan 2025 to Mar 2025"
_STATEMENT_DATE_RE = re.compile(
    r"statement\s+date\s*:?\s*"
    r"(\d{1,2}(?:st|nd|rd|th)?[\s/\-]+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:[\s/\-]+\d{2,4})?)",
    re.IGNORECASE,
)

# Textual date range: "For 1st June  30th June 2015"
_PERIOD_TEXT_RE = re.compile(
    r"(?:for|from)\s+"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)"
    r"\s+"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{4})?)",
    re.IGNORECASE,
)

# "covers the last 60 days of transactions: (15-May-2017to 27-Jul-2017)"
_PERIOD_COVERS_RE = re.compile(
    r"covers\s+.*?"
    r"\(?\s*(\d{1,2}[/\-](?:\w+)[/\-]\d{2,4})\s*"
    r"(?:to)?\s*"
    r"(\d{1,2}[/\-](?:\w+)[/\-]\d{2,4})\s*\)?",
    re.IGNORECASE,
)

# Generic two-date range with "to" separator
_GENERIC_DATE_RANGE_RE = re.compile(
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+"
    r"to\s+"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────
#  Date format patterns for header dates
# ──────────────────────────────────────────────

_DATE_RE = re.compile(
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{4})?",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _header_tokens(
    tokens: List[OCRToken],
    fraction: float = 0.35,
) -> List[OCRToken]:
    """Return tokens from the top *fraction* of the page.

    We estimate page height from the maximum y-coordinate in the
    bounding boxes, then keep only tokens whose top edge (y_min) is
    in the upper ``fraction`` of that range.
    """
    if not tokens:
        return []

    max_y = max(t.bbox[3] for t in tokens)
    cutoff = max_y * fraction
    return [t for t in tokens if t.bbox[1] <= cutoff]


def _concat_text(tokens: List[OCRToken], sep: str = " ") -> str:
    """Concatenate token texts in reading order (as given)."""
    return sep.join(t.text for t in tokens)


def _avg_confidence(tokens: List[OCRToken]) -> float:
    """Mean confidence across a set of tokens."""
    if not tokens:
        return 0.0
    return sum(t.confidence for t in tokens) / len(tokens)


def _find_tokens_matching(
    tokens: List[OCRToken],
    pattern: str,
) -> List[OCRToken]:
    """Return tokens whose text contains *pattern* (case-insensitive)."""
    pat = pattern.lower()
    return [t for t in tokens if pat in t.text.lower()]


def _clean_account_number(raw: str) -> str:
    """Normalize an extracted account number string."""
    # Strip surrounding whitespace and common OCR artifacts
    cleaned = raw.strip().strip(":").strip()
    # Collapse internal spaces
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ──────────────────────────────────────────────
#  Individual field extractors
# ──────────────────────────────────────────────

def _extract_bank_name(
    header: List[OCRToken],
) -> Tuple[Optional[str], float]:
    """Detect bank name from header tokens.

    Strategy: check if any known bank keyword appears as a substring
    in the concatenated header text.  Longest match wins.  Confidence
    is the average OCR confidence of the tokens that contain the match.
    """
    header_text = _concat_text(header).lower()

    for keyword, canonical in _BANK_KEYWORDS_SORTED:
        if keyword in header_text:
            # Find matching tokens for confidence estimation
            matching = _find_tokens_matching(header, keyword)
            conf = _avg_confidence(matching) if matching else 0.5
            # Scale slightly: exact substring match is fairly reliable
            conf = _clamp_confidence(conf * 0.9 + 0.1)
            logger.debug("Bank detected: %s (conf=%.4f)", canonical, conf)
            return canonical, round(conf, 4)

    return None, 0.0


def _extract_account_number(
    header: List[OCRToken],
) -> Tuple[Optional[str], float]:
    """Extract account number from header tokens.

    Looks for patterns like:
      - "Account Number: 201301401"
      - "Account No :910010041701828"
      - "Statement of Account: 2518XXXXXXXXO946"
      - "A/C No: 12345678"
    """
    header_text = _concat_text(header)

    # Try "Statement of Account" pattern first (often includes masked numbers)
    m = _STATEMENT_ACCOUNT_RE.search(header_text)
    if m:
        acct = _clean_account_number(m.group(1))
        if len(acct) >= 4:
            matching = _find_tokens_matching(header, acct[:6])
            conf = _avg_confidence(matching) if matching else 0.5
            conf = _clamp_confidence(conf * 0.85 + 0.1)
            return acct, round(conf, 4)

    # Try standard "Account Number/No" label
    m = _ACCOUNT_LABEL_RE.search(header_text)
    if m:
        acct = _clean_account_number(m.group(1))
        if len(acct) >= 4:
            matching = _find_tokens_matching(header, acct[:6])
            conf = _avg_confidence(matching) if matching else 0.5
            conf = _clamp_confidence(conf * 0.85 + 0.1)
            return acct, round(conf, 4)

    return None, 0.0


def _extract_account_holder(
    header: List[OCRToken],
) -> Tuple[Optional[str], float]:
    """Extract account holder name from header tokens.

    Strategy:
      1. Look for explicit label: "Customer Name:", "Account Holder:", etc.
      2. If no label found, heuristic: the first multi-word token in the
         header zone (after the bank name region, before account info)
         that looks like a person's name (all-alpha words, title case).
    """
    header_text = _concat_text(header)

    # Strategy 1: labeled extraction
    m = _HOLDER_LABEL_RE.search(header_text)
    if m:
        name = m.group(1).strip()
        # Clean common trailing garbage
        name = re.split(r"\b(?:address|city|pin|nominee|a/c|account)\b",
                        name, flags=re.IGNORECASE)[0].strip()
        # Remove trailing punctuation / digits
        name = re.sub(r"[:\d]+$", "", name).strip()
        if len(name) >= 2:
            matching = _find_tokens_matching(header, name[:10])
            conf = _avg_confidence(matching) if matching else 0.5
            conf = _clamp_confidence(conf * 0.85 + 0.1)
            return name, round(conf, 4)

    # Strategy 2: heuristic — find name-like tokens in header
    # Look for tokens that are all-alpha words (2+ words), appearing
    # early in the page, and not matching known keywords.
    skip_words = {
        "bank", "statement", "account", "branch", "date", "balance",
        "opening", "closing", "transactions", "details", "information",
        "current", "savings", "period", "page", "from", "address",
        "customer", "number", "currency", "deposit", "withdrawal",
        "credit", "debit", "summary", "the", "for", "with",
        # Bank product names (avoid false-positive holder matches)
        "choice", "classic", "premium", "platinum", "gold", "silver",
        "plus", "select", "power", "privilege", "salary", "advantage",
        "recent", "banking",
    }

    for token in header:
        text = token.text.strip()
        words = text.split()
        # Name heuristic: 2-4 words, mostly alpha, reasonable length
        if 2 <= len(words) <= 5 and all(w.isalpha() for w in words):
            if not any(w.lower() in skip_words for w in words):
                conf = _clamp_confidence(token.confidence * 0.5 + 0.1)
                return text, round(conf, 4)

    return None, 0.0


def _extract_statement_period(
    header: List[OCRToken],
) -> Tuple[Optional[str], Optional[str], float]:
    """Extract statement period (from, to) from header tokens.

    Returns (period_from, period_to, confidence).
    """
    header_text = _concat_text(header)

    # Strategy 1: explicit "For Period: date to date"
    m = _PERIOD_EXPLICIT_RE.search(header_text)
    if m:
        matching = _find_tokens_matching(header, "period")
        conf = _avg_confidence(matching) if matching else 0.5
        conf = _clamp_confidence(conf * 0.85 + 0.1)
        return m.group(1), m.group(2), round(conf, 4)

    # Strategy 2: "(From date  date)"
    m = _PERIOD_FROM_TO_RE.search(header_text)
    if m:
        matching = _find_tokens_matching(header, "from")
        conf = _avg_confidence(matching) if matching else 0.5
        conf = _clamp_confidence(conf * 0.85 + 0.1)
        return m.group(1), m.group(2), round(conf, 4)

    # Strategy 3: "covers ... (date to date)"
    m = _PERIOD_COVERS_RE.search(header_text)
    if m:
        matching = _find_tokens_matching(header, "cover")
        conf = _avg_confidence(matching) if matching else 0.4
        conf = _clamp_confidence(conf * 0.80 + 0.1)
        return m.group(1), m.group(2), round(conf, 4)

    # Strategy 4: textual dates "For 1st June  30th June 2015"
    m = _PERIOD_TEXT_RE.search(header_text)
    if m:
        matching = _find_tokens_matching(header, "for")
        conf = _avg_confidence(matching) if matching else 0.4
        conf = _clamp_confidence(conf * 0.70 + 0.1)
        return m.group(1).strip(), m.group(2).strip(), round(conf, 4)

    # Strategy 5: generic "date to date"
    m = _GENERIC_DATE_RANGE_RE.search(header_text)
    if m:
        conf = 0.35  # Low confidence — no label context
        return m.group(1), m.group(2), conf

    # Strategy 6: single statement date
    m = _STATEMENT_DATE_RE.search(header_text)
    if m:
        date_text = m.group(1).strip()
        matching = _find_tokens_matching(header, "statement date")
        conf = _avg_confidence(matching) if matching else 0.3
        conf = _clamp_confidence(conf * 0.6 + 0.1)
        return None, date_text, round(conf, 4)

    return None, None, 0.0


def _extract_currency(
    header: List[OCRToken],
    all_tokens: List[OCRToken],
) -> Tuple[Optional[str], float]:
    """Detect currency from header tokens, falling back to all tokens.

    Strategy:
      1. Check for explicit "Currency: XXX" label in header.
      2. Check for currency symbols (₹, $, £, €) in header.
      3. Fall back to checking all tokens for symbols.
    """
    # Search header first, then all tokens as fallback
    for search_scope, scope_penalty in [(header, 1.0), (all_tokens, 0.7)]:
        scope_text = _concat_text(search_scope)

        for pattern, fixed_code in _CURRENCY_PATTERNS:
            m = pattern.search(scope_text)
            if m:
                if fixed_code is None:
                    # Dynamic: extract code from regex group
                    code = m.group(1).upper()
                    # Validate it looks like an ISO code
                    if len(code) == 3 and code.isalpha():
                        # Handle common OCR errors
                        code = _fix_currency_ocr(code)
                        matching = _find_tokens_matching(search_scope, "currency")
                        conf = _avg_confidence(matching) if matching else 0.5
                        conf = _clamp_confidence(conf * 0.85 * scope_penalty + 0.1)
                        return code, round(conf, 4)
                else:
                    matching = [
                        t for t in search_scope
                        if pattern.search(t.text)
                    ]
                    conf = _avg_confidence(matching) if matching else 0.5
                    conf = _clamp_confidence(conf * 0.80 * scope_penalty + 0.1)
                    return fixed_code, round(conf, 4)

    return None, 0.0


def _fix_currency_ocr(code: str) -> str:
    """Fix common OCR mis-reads in currency codes."""
    fixes = {
        "LNR": "INR",  # common OCR misread: I → L
        "1NR": "INR",
        "USO": "USD",
        "USC": "USD",
        "GBR": "GBP",
        "EUP": "EUR",
    }
    return fixes.get(code, code)


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def extract_metadata(tokens: List[OCRToken]) -> ExtractionResult:
    """Extract statement metadata from a list of OCR tokens.

    This is the main entry point.  It performs a **two-pass search**:

      1. **Header pass** — search only the top ~35 % of the page (where
         metadata typically lives).
      2. **Full-page fallback** — for any fields *not* extracted in pass 1,
         re-run the extractors against *all* tokens with a confidence
         penalty, since matches outside the header zone are less reliable.

    This handles layouts like PNB statements where branch/customer details
    push the account summary to the middle of the page.

    Args:
        tokens: Complete list of :class:`OCRToken` objects from a single
                statement page, in reading order.

    Returns:
        An :class:`ExtractionResult` with metadata and confidence scores.
    """
    if not tokens:
        logger.warning("No tokens provided for metadata extraction.")
        return ExtractionResult()

    header = _header_tokens(tokens)
    logger.debug(
        "Header zone: %d / %d tokens (%.0f%%)",
        len(header), len(tokens),
        100 * len(header) / len(tokens) if tokens else 0,
    )

    result = ExtractionResult()

    # Confidence penalty applied when a field is found outside the header
    _FALLBACK_PENALTY = 0.80

    # ── Pass 1: Header zone ───────────────
    bank, bank_conf = _extract_bank_name(header)
    result.metadata.bank_name = bank
    result.confidence["bank_name"] = bank_conf

    acct, acct_conf = _extract_account_number(header)
    result.metadata.account_number = acct
    result.confidence["account_number"] = acct_conf

    holder, holder_conf = _extract_account_holder(header)
    result.metadata.account_holder = holder
    result.confidence["account_holder"] = holder_conf

    period_from, period_to, period_conf = _extract_statement_period(header)
    result.metadata.statement_period_from = period_from
    result.metadata.statement_period_to = period_to
    result.confidence["statement_period_from"] = period_conf
    result.confidence["statement_period_to"] = period_conf

    currency, currency_conf = _extract_currency(header, tokens)
    result.metadata.currency = currency
    result.confidence["currency"] = currency_conf

    # ── Pass 2: Full-page fallback for missed fields ──
    if result.metadata.bank_name is None:
        bank, bank_conf = _extract_bank_name(tokens)
        if bank:
            result.metadata.bank_name = bank
            result.confidence["bank_name"] = round(bank_conf * _FALLBACK_PENALTY, 4)

    if result.metadata.account_number is None:
        acct, acct_conf = _extract_account_number(tokens)
        if acct:
            result.metadata.account_number = acct
            result.confidence["account_number"] = round(acct_conf * _FALLBACK_PENALTY, 4)

    if result.metadata.account_holder is None:
        holder, holder_conf = _extract_account_holder(tokens)
        if holder:
            result.metadata.account_holder = holder
            result.confidence["account_holder"] = round(holder_conf * _FALLBACK_PENALTY, 4)

    if result.metadata.statement_period_from is None and result.metadata.statement_period_to is None:
        period_from, period_to, period_conf = _extract_statement_period(tokens)
        if period_from or period_to:
            result.metadata.statement_period_from = period_from
            result.metadata.statement_period_to = period_to
            penalized = round(period_conf * _FALLBACK_PENALTY, 4)
            result.confidence["statement_period_from"] = penalized
            result.confidence["statement_period_to"] = penalized

    logger.info(
        "Extraction complete: bank=%s, holder=%s, acct=%s, "
        "period=%s→%s, currency=%s",
        result.metadata.bank_name,
        result.metadata.account_holder,
        result.metadata.account_number,
        result.metadata.statement_period_from,
        result.metadata.statement_period_to,
        result.metadata.currency,
    )

    return result
