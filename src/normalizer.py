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
    
    OCR_DATE_SUBSTITUTIONS = [
        # Digit-letter confusion in day/year positions
        ("1U", "10"), ("lU", "10"), ("IU", "10"),
        ("1O", "10"), ("lO", "10"),   # letter-O vs zero
        ("0l", "01"), ("0I", "01"),   # letter-l/I vs digit-1
        ("ZU2L", "2021"), ("ZU2l", "2021"), ("2U2l", "2021"),
        ("202L", "2021"), ("2O2L", "2021"),
        ("4k24", "2024"), ("0x4L", "2024"), ("0AL204", "2024"),
        ("l9", "19"), ("I9", "19"), ("1l", "11"),
        # Separator confusion
        ("@", "0"),
        # Year-end corruptions
        ("2W2", "202"), ("202l", "2021"),
    ]

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

    for bad, good in OCR_DATE_SUBSTITUTIONS:
        date_str = date_str.replace(bad, good)

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
            
    # Add sequential date interpolation
    from dateutil import parser as date_parser
    import datetime
    
    def parse_date_safely(date_str: str) -> datetime.date | None:
        try:
            clean_str = re.sub(r"\s+", " ", date_str).strip()
            dt = date_parser.parse(clean_str, fuzzy=True)
            return dt.date()
        except Exception:
            return None

    interpolated_count = 0
    n = len(rows)
    for i in range(1, n - 1):
        if DATE_RE.search(rows[i].text) is not None:
            continue
            
        prev_match = DATE_RE.search(rows[i - 1].text)
        next_match = DATE_RE.search(rows[i + 1].text)
        if not prev_match or not next_match:
            continue
            
        prev_date_str = prev_match.group(0)
        next_date_str = next_match.group(0)
        
        d1 = parse_date_safely(prev_date_str)
        d2 = parse_date_safely(next_date_str)
        if not d1 or not d2:
            continue
            
        diff = (d2 - d1).days
        interpolated_date = None
        if abs(diff) == 1:
            interpolated_date = d1
        elif abs(diff) == 2:
            interpolated_date = d1 + datetime.timedelta(days=1 if diff > 0 else -1)
        elif abs(diff) == 0:
            interpolated_date = d1
            
        if interpolated_date:
            has_year = bool(re.search(r"\b\d{4}\b", prev_date_str + next_date_str))
            has_letters = any(c.isalpha() for c in prev_date_str + next_date_str)
            if has_letters:
                if has_year:
                    interpolated_str = interpolated_date.strftime("%d %B %Y")
                else:
                    interpolated_str = interpolated_date.strftime("%d %B")
            else:
                if has_year:
                    interpolated_str = interpolated_date.strftime("%d-%m-%Y")
                else:
                    sep = "-"
                    if "/" in prev_date_str or "/" in next_date_str:
                        sep = "/"
                    elif "." in prev_date_str or "." in next_date_str:
                        sep = "."
                    interpolated_str = interpolated_date.strftime(f"%d{sep}%m")
            
            old_text = rows[i].text
            rows[i].text = f"{interpolated_str} {old_text}"
            interpolated_count += 1
            logger.debug(
                "Interpolated date %s for row %d: %r -> %r",
                interpolated_str,
                i,
                old_text,
                rows[i].text
            )
            
    normalized_date_count += interpolated_count
    return rows, normalized_date_count


# ──────────────────────────────────────────────
#  Amount Normalization
# ──────────────────────────────────────────────

# Import MONEY_RE and DATE_RE from the transaction parser/constants.
from src.constants import MONEY_RE, DATE_RE

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
    """Apply the full normalization pipeline to a single amount candidate."""
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
    s = re.sub(r"(\d),(\d{2})$", r"\1.\2", s)

    # Step 5: Normalize whitespace-as-decimal
    s = re.sub(r"(\d)\s+(\d{2})$", r"\1.\2", s)

    return s


def _repair_amount_separators(amount_str: str) -> str:
    """Applies adaptive amount separator repair rules in order.
    
    1. Missing decimal: if the string matches r"^\-?\d{1,6}$", append ".00" (if <= 7 digits before decimal).
    2. Letter-zero confusion in decimals: replace terminal "C0", "OO", "Co" with "00".
    3. Non-digit noise prefix: strip leading chars that are not digits, "-", or "+" from the start.
       Apply letter cleaning: replace "n", "o", "O" with "0".
    4. Validate against MONEY_RE (ignoring signs).
    """
    if not any(c.isdigit() for c in amount_str):
        return amount_str
        
    original = amount_str
    
    # Rule 1: Missing decimal
    if re.match(r"^\-?\d{1,6}$", amount_str) or re.match(r"^\-?\d{1,7}$", amount_str):
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
        
        # Clean integer part
        sign = ""
        if integer_part and integer_part[0] in "-+":
            sign = integer_part[0]
            integer_part = integer_part[1:]
        integer_part = "".join(c for c in integer_part if c.isdigit())
        
        # Clean decimal part
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


def normalize_amount_strings(rows: List[Row]) -> Tuple[List[Row], int]:
    """Apply amount normalization to row texts."""
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
            logger.debug(
                "Amount-normalized row: %r → %r",
                old_text,
                new_text,
            )

    return rows, normalized_amount_count
