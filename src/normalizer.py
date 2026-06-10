"""
normalizer.py - Dedicated normalization layer for date and amount parsing robustness.

Fixes common OCR errors in dates and monetary values, normalizes ordinal
day formats and corrupted decimal separators, without mutating text globally.

Amount normalization rules (applied only to amount-candidate spans):
  1. Comma decimal separator:  925,90 → 925.90
  2. OCR character substitution:  530.C0 → 530.00, 9200,C0 → 9200.00
  3. Whitespace decimal separator:  -350 00 → -350.00
  4. Leading noise stripping:  ~8300.00 → 8300.00, $24.50 → 24.50
  5. Duplicated/trailing separators:  92.,50 → 92.50
All substitutions are validated against MONEY_RE after application.
If the result is not a valid amount, the change is discarded.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)

MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# Matches ordinal suffixes attached to 1-2 digits, followed by a month name.
ORDINAL_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th|uh|ih|\")\s+(?=(?:{MONTHS}))",
    re.IGNORECASE
)

# Matches date-like phrases that might contain OCR errors.
# Supports formats like DD/MM/YYYY, DD-MMM-YYYY, DD MMM YYYY, etc.
# with known OCR typos included in the pattern so it matches.
DATE_LIKE_RE = re.compile(
    r"\b"
    r"(?:\d{1,2}|0J|l9|I9|1l)"                # Day
    r"(?:st|nd|rd|th|uh|ih|\"|h)?"            # Ordinal (often mangled)
    r"[\s/\-\.]+"                             # Separator (including dot like 13.10)
    r"(?:\d{1,2}|" + MONTHS + r")[a-z]*"      # Month
    r"(?:[\s/\-\.]+(?:\d{2,4}|2O2I|\?017))?"  # Year
    r"\b",
    re.IGNORECASE
)


def _correct_ocr_typos(date_str: str) -> str:
    """Applies specific OCR corrections to a string known to be a date."""
    original = date_str
    
    # Ordinal removal (in case they weren't caught by ORDINAL_RE due to mangling inside slashes)
    date_str = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th|uh|ih|\"|h)\b", r"\1", date_str, flags=re.IGNORECASE)
    
    # Specific targeted typos
    date_str = date_str.replace("0J", "01")
    date_str = date_str.replace("2O2I", "2021")
    date_str = date_str.replace("l9", "19")
    date_str = date_str.replace("I9", "19")
    date_str = date_str.replace("1l", "11")
    date_str = date_str.replace("?017", "2017")
    
    # Fix dot separators that should be hyphens or slashes (e.g., 13.10-2021)
    date_str = re.sub(r"(\d{1,2})\.(\d{1,2})[\.\-](\d{2,4})", r"\1-\2-\3", date_str)

    return date_str


def normalize_date_strings(rows: List[Row]) -> Tuple[List[Row], int]:
    """
    Applies date normalization to row texts.
    
    Returns:
        A tuple of (modified_rows, normalized_date_count).
    """
    normalized_date_count = 0
    
    for row in rows:
        old_text = row.text
        new_text = old_text
        
        # 1. Normalize explicit ordinal dates (1st June -> 1 June)
        new_text = ORDINAL_RE.sub(r"\1 ", new_text)
        
        # 2. Find date-like spans and correct OCR typos inside them only
        def _replace_match(match: re.Match) -> str:
            return _correct_ocr_typos(match.group(0))
            
        new_text = DATE_LIKE_RE.sub(_replace_match, new_text)
        
        # We also want to catch things like `Ist June` or `ghJune` from failure reports.
        # "Ist" -> "1st" -> "1"
        new_text = re.sub(rf"\bIst\s+(?=(?:{MONTHS}))", "1 ", new_text, flags=re.IGNORECASE)
        # "ghJune" -> "9th June" -> "9 June"
        new_text = re.sub(rf"\bgh\s*(?=(?:{MONTHS}))", "9 ", new_text, flags=re.IGNORECASE)
        
        if new_text != old_text:
            row.text = new_text
            normalized_date_count += 1
            
    return rows, normalized_date_count


# ──────────────────────────────────────────────
#  Amount Normalization
# ──────────────────────────────────────────────

# Import MONEY_RE from the transaction parser — this is the canonical
# definition of what constitutes a "valid monetary value" downstream.
# We validate all normalization results against it.
from src.transaction_parser import MONEY_RE

# Matches tokens that *resemble* monetary values but may contain OCR
# corruption.  Designed to be broader than MONEY_RE so we can catch
# corrupted strings, clean them, and then validate against MONEY_RE.
#
# Captures groups of digits interspersed with:
#   - commas (thousands or European decimal)
#   - periods (decimal point)
#   - OCR substitution characters: O, o, C, l, I, + (when adjacent to digits)
#   - leading noise: ~, $, ₹, €, £
#   - whitespace gaps inside numeric sequences (e.g. "-350 00")
#
# The pattern is intentionally generous — false positives are filtered out
# by the MONEY_RE validation gate after cleaning.
AMOUNT_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z])"                    # Must not be preceded by a letter
    r"[~$₹€£]?"                         # Optional leading currency/noise
    r"-?"                               # Optional negative sign
    r"\d[\d,.\sOoCcIl+]*\d"            # Core: digits with corruption chars
    r"(?![A-Za-z]{2})"                  # Must not be followed by 2+ letters (avoid dates like "2017Jul")
)

# Characters that OCR commonly substitutes for digits in amount contexts.
# Each tuple: (pattern_char, replacement_digit, context_description)
_OCR_DIGIT_MAP = {
    "O": "0",
    "o": "0",
    "C": "0",
    "c": "0",
    "I": "1",
    "l": "1",
    "+": "4",
}


def _normalize_amount_token(token: str) -> str:
    """Apply the full normalization pipeline to a single amount candidate.

    Steps:
      1. Strip leading noise characters (~, $, ₹, €, £)
      2. Apply OCR character substitutions (O→0, C→0, l→1, etc.)
      3. Normalize comma-as-decimal: 925,90 → 925.90
      4. Normalize whitespace-as-decimal: -350 00 → -350.00
      5. Fix duplicated/misplaced separators: 92.,50 → 92.50

    Returns the cleaned string.  Caller must validate against MONEY_RE.
    """
    s = token.strip()

    # Step 1: Strip leading noise
    while s and s[0] in "~$₹€£":
        s = s[1:]
    s = s.strip()

    if not s:
        return s

    # Step 2: OCR character substitutions
    # Only substitute characters that are surrounded by digit-like context
    # to avoid corrupting actual text.
    result = []
    for i, ch in enumerate(s):
        if ch in _OCR_DIGIT_MAP:
            # Check if this character is adjacent to at least one digit
            prev_is_digit = (i > 0 and (s[i - 1].isdigit() or s[i - 1] in ".,"))
            next_is_digit = (i < len(s) - 1 and (s[i + 1].isdigit() or s[i + 1] in ".,"))
            if prev_is_digit or next_is_digit:
                result.append(_OCR_DIGIT_MAP[ch])
            else:
                result.append(ch)
        else:
            result.append(ch)
    s = "".join(result)

    # Step 3: Fix duplicated/misplaced separators like "92.,50" → "92.50"
    s = s.replace(".,", ".")
    s = s.replace(",.", ".")

    # Step 4: Normalize comma-as-decimal-separator
    # Pattern: digits followed by comma then exactly 2 digits at end of token
    # e.g. "925,90" → "925.90", "5987,20" → "5987.20"
    # But NOT "1,234,567" (thousands separators)
    s = re.sub(r"(\d),(\d{2})$", r"\1.\2", s)

    # Step 5: Normalize whitespace-as-decimal
    # e.g. "-350 00" → "-350.00"
    s = re.sub(r"(\d)\s+(\d{2})$", r"\1.\2", s)

    return s


def normalize_amount_strings(rows: List[Row]) -> Tuple[List[Row], int]:
    """Apply amount normalization to row texts.

    For each row, finds amount-candidate spans and attempts normalization.
    Only commits the change if the normalized text produces a valid
    MONEY_RE match that didn't exist before.

    Returns:
        A tuple of (modified_rows, normalized_amount_count).
        normalized_amount_count is the number of *rows* whose text was
        modified (not the number of individual amounts fixed).
    """
    normalized_amount_count = 0

    for row in rows:
        old_text = row.text

        # Check what MONEY_RE already matches before normalization
        existing_amounts = set(MONEY_RE.findall(old_text))

        def _replace_amount_candidate(match: re.Match) -> str:
            original_span = match.group(0)
            normalized = _normalize_amount_token(original_span)

            # Only use the normalization if it produces a new valid amount
            # that MONEY_RE recognizes and that didn't already exist
            new_amounts = MONEY_RE.findall(normalized)
            if new_amounts and not all(a in existing_amounts for a in new_amounts):
                return normalized

            # Also accept if the normalized form IS a valid amount itself
            # (the whole candidate becomes one clean amount)
            if MONEY_RE.fullmatch(normalized):
                return normalized

            # Normalization didn't help — keep original
            return original_span

        new_text = AMOUNT_CANDIDATE_RE.sub(_replace_amount_candidate, old_text)

        if new_text != old_text:
            row.text = new_text
            normalized_amount_count += 1
            logger.debug(
                "Amount-normalized row: %r → %r",
                old_text,
                new_text,
            )

    return rows, normalized_amount_count

