"""
normalizers/amount_normalizer.py — Monetary amount normalization for OCR output.

Fixes common OCR errors in monetary value strings:
  1. Comma decimal separator:  925,90 → 925.90
  2. OCR character substitution:  530.C0 → 530.00, 9200,C0 → 9200.00
  3. Whitespace decimal separator:  -350 00 → -350.00
  4. Leading noise stripping:  ~8300.00 → 8300.00, $24.50 → 24.50
  5. Duplicated/trailing separators:  92.,50 → 92.50

All substitutions are validated against MONEY_RE after application.
If the result is not a valid amount the change is discarded.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Matches tokens that *resemble* monetary values but may contain OCR
# corruption.  Designed to be broader than MONEY_RE so we can catch
# corrupted strings, clean them, and then validate against MONEY_RE.
AMOUNT_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z])"                    # Must not be preceded by a letter
    r"[~$₹€£]?"                         # Optional leading currency/noise
    r"-?"                               # Optional negative sign
    r"\d[\d,.\sOoCcIl+]*\d"            # Core: digits with corruption chars
    r"(?![A-Za-z]{2})"                  # Must not be followed by 2+ letters
)

# Characters that OCR commonly substitutes for digits in amount contexts.
_OCR_DIGIT_MAP: dict[str, str] = {
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

    Args:
        token: Raw amount string potentially containing OCR errors.

    Returns:
        Cleaned amount string.
    """
    s = token.strip()

    # Step 1: Strip leading noise
    while s and s[0] in "~$₹€£":
        s = s[1:]
    s = s.strip()

    if not s:
        return s

    # Step 2: OCR character substitutions
    result = []
    for i, ch in enumerate(s):
        if ch in _OCR_DIGIT_MAP:
            prev_is_digit = i > 0 and (s[i - 1].isdigit() or s[i - 1] in ".,")
            next_is_digit = i < len(s) - 1 and (s[i + 1].isdigit() or s[i + 1] in ".,")
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
    s = re.sub(r"(\d),(\d{2})$", r"\1.\2", s)

    # Step 5: Normalize whitespace-as-decimal
    s = re.sub(r"(\d)\s+(\d{2})$", r"\1.\2", s)

    return s


def _repair_amount_separators(amount_str: str) -> str:
    """Apply adaptive amount separator repair rules.

    Rules applied in order:
      1. Missing decimal: if the string is a bare integer, append ".00".
      2. Letter-zero confusion in decimals: replace terminal "C0"/"OO"/"Co"
         with "00".
      3. Non-digit noise prefix: strip leading chars that are not digits,
         "-", or "+".  Apply letter cleaning: replace "n"/"o"/"O" with "0".
      4. Validate against MONEY_RE.

    Args:
        amount_str: Partially cleaned amount string.

    Returns:
        Repaired amount string, or the original if repair fails validation.
    """
    from src.constants import MONEY_RE

    if not any(c.isdigit() for c in amount_str):
        return amount_str

    original = amount_str

    # Rule 1: Missing decimal
    if re.match(r"^\-?\d{1,7}$", amount_str):
        amount_str = amount_str + ".00"

    # Rule 2: Letter-zero confusion in decimals
    if amount_str.endswith("C0") or amount_str.endswith("OO") or amount_str.endswith("Co"):
        amount_str = amount_str[:-2] + "00"

    # Rule 3: Non-digit noise prefix and letter cleaning
    amount_str = amount_str.lstrip()
    while amount_str and not (amount_str[0].isdigit() or amount_str[0] in "-+"):
        amount_str = amount_str[1:]

    amount_str = amount_str.replace("n", "0").replace("o", "0").replace("O", "0")

    # Separator fix
    seps = [i for i, c in enumerate(amount_str) if c in "., "]
    if seps:
        last_sep_idx = seps[-1]
        integer_part = amount_str[:last_sep_idx]
        decimal_part = amount_str[last_sep_idx + 1:]

        sign = ""
        if integer_part and integer_part[0] in "-+":
            sign = integer_part[0]
            integer_part = integer_part[1:]
        integer_part = "".join(c for c in integer_part if c.isdigit())

        decimal_part = "".join(c for c in decimal_part if c.isdigit())
        if len(decimal_part) > 2:
            decimal_part = decimal_part[:2]
        elif len(decimal_part) < 2:
            decimal_part = decimal_part.ljust(2, "0")

        if integer_part:
            formatted_int = f"{int(integer_part):,}"
            amount_str = f"{sign}{formatted_int}.{decimal_part}"
        else:
            amount_str = f"{sign}0.{decimal_part}"

    # Rule 4: Validate against MONEY_RE (ignoring optional leading sign)
    val_str = amount_str
    if val_str.startswith("-") or val_str.startswith("+"):
        val_str = val_str[1:]

    if MONEY_RE.fullmatch(val_str):
        return amount_str
    return original


def normalize_amount_strings(rows) -> Tuple[list, int]:
    """Apply amount normalization to a list of Row objects.

    Args:
        rows: List of :class:`~src.transaction_reconstructor.Row` objects.

    Returns:
        A tuple of ``(modified_rows, normalized_amount_count)``.
    """
    from src.constants import MONEY_RE

    normalized_amount_count = 0

    for row in rows:
        old_text = row.text
        existing_amounts = set(MONEY_RE.findall(old_text))

        def _replace_amount_candidate(match: re.Match) -> str:
            original_span = match.group(0)
            normalized = _normalize_amount_token(original_span)
            normalized = _repair_amount_separators(normalized)

            new_amounts = MONEY_RE.findall(normalized)
            if new_amounts and not all(a in existing_amounts for a in new_amounts):
                return normalized

            if MONEY_RE.fullmatch(normalized):
                return normalized

            return original_span

        new_text = AMOUNT_CANDIDATE_RE.sub(_replace_amount_candidate, old_text)

        if new_text != old_text:
            row.text = new_text
            normalized_amount_count += 1
            logger.debug("Amount-normalized row: %r → %r", old_text, new_text)

    return rows, normalized_amount_count