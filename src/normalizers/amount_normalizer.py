"""
normalizers/amount_normalizer.py — Amount normalization for OCR-extracted text.

Fixes common OCR corruption in monetary value strings and normalizes
non-standard separators.  Corrections are applied only to amount-candidate
spans; other text is left untouched.

Normalization rules (applied per amount-candidate span):
  1. Strip leading currency/noise characters:  ~8300.00 → 8300.00
  2. OCR character substitutions in numeric context:
       530.C0 → 530.00,  9200,C0 → 9200.00
  3. Fix duplicated/misplaced separators:  92.,50 → 92.50
  4. Comma-as-decimal-separator:  925,90 → 925.90
  5. Whitespace-as-decimal:  -350 00 → -350.00

After each transformation the result is validated against MONEY_RE.
If validation fails, the original span is preserved.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from src.constants import MONEY_RE
from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)

# Broader than MONEY_RE — catches corrupted amount strings so they can be
# cleaned and then validated.
AMOUNT_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z])"           # Must not be preceded by a letter
    r"[~$₹€£]?"                # Optional leading currency/noise
    r"-?"                      # Optional negative sign
    r"\d[\d,.\sOoCcIl+]*\d"   # Core: digits with common corruption chars
    r"(?![A-Za-z]{2})"         # Must not be followed by 2+ letters
)

# OCR characters that should be treated as digits when surrounded by digits
# or decimal/thousands separators.
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
    """Apply the core normalization pipeline to a single amount candidate.

    Steps:
      1. Strip leading currency/noise characters.
      2. Replace OCR letter-digit confusion characters when flanked by
         numeric context.
      3. Fix duplicated/misplaced separators (``.,`` → ``.``).
      4. Normalize trailing comma-decimal (``925,90`` → ``925.90``).
      5. Normalize trailing whitespace-decimal (``350 00`` → ``350.00``).

    Args:
        token: A raw amount candidate string.

    Returns:
        The cleaned string.  If no changes apply the original is returned.
    """
    s = token.strip()

    # Step 1: strip leading noise
    while s and s[0] in "~$₹€£":
        s = s[1:]
    s = s.strip()
    if not s:
        return s

    # Step 2: OCR character substitution (context-sensitive)
    result = []
    for i, ch in enumerate(s):
        if ch in _OCR_DIGIT_MAP:
            prev_ok = i > 0 and (s[i - 1].isdigit() or s[i - 1] in ".,")
            next_ok = i < len(s) - 1 and (s[i + 1].isdigit() or s[i + 1] in ".,")
            result.append(_OCR_DIGIT_MAP[ch] if (prev_ok or next_ok) else ch)
        else:
            result.append(ch)
    s = "".join(result)

    # Step 3: fix duplicated/misplaced separators
    s = s.replace(".,", ".")
    s = s.replace(",.", ".")

    # Step 4: comma-as-decimal (only when followed by exactly 2 digits at end)
    s = re.sub(r"(\d),(\d{2})$", r"\1.\2", s)

    # Step 5: whitespace-as-decimal
    s = re.sub(r"(\d)\s+(\d{2})$", r"\1.\2", s)

    return s


def _repair_amount_separators(amount_str: str) -> str:
    """Apply adaptive separator repair rules to a partially-cleaned amount string.

    Rule 1: Append ``.00`` if the string is a plain integer (≤ 7 digits).
    Rule 2: Fix terminal letter-zero confusion (``C0``, ``OO``, ``Co`` → ``00``).
    Rule 3: Strip non-digit leading noise; replace residual ``n``/``o``/``O``
            with ``0``.
    Separator fix: reconstruct ``integer.decimal`` from the last separator.
    Rule 4: Validate the result against MONEY_RE; return original on failure.

    Args:
        amount_str: A token that has already been through _normalize_amount_token.

    Returns:
        The repaired string, or the original if repair would invalidate it.
    """
    if not any(c.isdigit() for c in amount_str):
        return amount_str

    original = amount_str

    # Rule 1: missing decimal point
    if re.match(r"^\-?\d{1,7}$", amount_str):
        amount_str = amount_str + ".00"

    # Rule 2: terminal letter-zero confusion
    if amount_str.endswith("C0") or amount_str.endswith("OO") or amount_str.endswith("Co"):
        amount_str = amount_str[:-2] + "00"

    # Rule 3: non-digit noise prefix and letter cleaning
    amount_str = amount_str.lstrip()
    while amount_str and not (amount_str[0].isdigit() or amount_str[0] in "-+"):
        amount_str = amount_str[1:]

    amount_str = amount_str.replace("n", "0").replace("o", "0").replace("O", "0")

    # Separator reconstruction
    seps = [i for i, c in enumerate(amount_str) if c in "., "]
    if seps:
        last_sep = seps[-1]
        int_part = amount_str[:last_sep]
        dec_part = amount_str[last_sep + 1:]

        sign = ""
        if int_part and int_part[0] in "-+":
            sign = int_part[0]
            int_part = int_part[1:]
        int_part = "".join(c for c in int_part if c.isdigit())

        dec_part = "".join(c for c in dec_part if c.isdigit())
        if len(dec_part) > 2:
            dec_part = dec_part[:2]
        elif len(dec_part) < 2:
            dec_part = dec_part.ljust(2, "0")

        if int_part:
            amount_str = f"{sign}{int(int_part):,}.{dec_part}"
        else:
            amount_str = f"{sign}0.{dec_part}"

    # Rule 4: validate
    val_str = amount_str.lstrip("-+")
    if MONEY_RE.fullmatch(val_str):
        return amount_str
    return original


def normalize_amount_strings(rows: List[Row]) -> Tuple[List[Row], int]:
    """Apply amount normalization to a list of reconstructed rows.

    For each row, candidate amount spans are detected by AMOUNT_CANDIDATE_RE,
    cleaned, and validated.  Only spans that produce a valid MONEY_RE match
    (or produce a new valid match not present in the original text) are
    substituted back.

    Mutates ``row.text`` in-place.

    Args:
        rows: Reconstructed row list from transaction_reconstructor.

    Returns:
        A tuple ``(rows, count)`` where *count* is the number of rows
        whose text was modified.
    """
    normalized_amount_count = 0

    for row in rows:
        old_text = row.text
        existing_amounts = set(MONEY_RE.findall(old_text))

        def _replace_candidate(match: re.Match) -> str:
            original_span = match.group(0)
            normalized = _normalize_amount_token(original_span)
            normalized = _repair_amount_separators(normalized)

            new_amounts = MONEY_RE.findall(normalized)
            if new_amounts and not all(a in existing_amounts for a in new_amounts):
                return normalized
            if MONEY_RE.fullmatch(normalized):
                return normalized
            return original_span

        new_text = AMOUNT_CANDIDATE_RE.sub(_replace_candidate, old_text)

        if new_text != old_text:
            row.text = new_text
            normalized_amount_count += 1
            logger.debug("Amount-normalized row: %r → %r", old_text, new_text)

    return rows, normalized_amount_count