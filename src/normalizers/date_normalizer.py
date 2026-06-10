"""
normalizers/date_normalizer.py — Date normalization for OCR-extracted text.

Fixes common OCR errors in date strings and normalizes ordinal day formats.
Operates exclusively on date-like spans within row text; other text is
left untouched.

Normalization steps applied per row:
  1. Strip ordinal suffixes from day numbers before a month name
     (e.g. "1st June" → "1 June", "2nd Apr" → "2 Apr")
  2. Locate date-like spans and apply character-level OCR corrections
     inside those spans only.
  3. Fix specific mis-read prefixes ("Ist June" → "1 June",
     "ghJune" → "9 June").
  4. Sequential date interpolation: if a row lacks a date but both
     its neighbours have parseable dates that are within 2 days of
     each other, prepend an interpolated date to the row text.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import List, Tuple

from src.constants import DATE_RE
from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)

MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# Matches ordinal suffixes attached to 1-2 digits followed by a month name.
ORDINAL_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th|uh|ih|\")\\s+(?=(?:{MONTHS}))",
    re.IGNORECASE,
)

# Matches date-like phrases that may contain OCR errors.
# Broad enough to capture corrupted date strings so corrections
# can be applied inside the span before the canonical DATE_RE fires.
DATE_LIKE_RE = re.compile(
    r"\b"
    r"(?:\d{1,2}|0J|l9|I9|1l)"                 # Day (with known OCR variants)
    r"(?:st|nd|rd|th|uh|ih|\"|h)?"              # Optional ordinal suffix
    r"[\s/\-\.]+"                               # Separator (including dot)
    r"(?:\d{1,2}|" + MONTHS + r")[a-z]*"        # Month
    r"(?:[\s/\-\.]+(?:\d{2,4}|2O2I|\?017))?"   # Optional year
    r"\b",
    re.IGNORECASE,
)

# Lookup table of character-level OCR corrections applied *only* within
# date-like spans.  Ordered so longer patterns are tried before substrings.
_OCR_DATE_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("1U", "10"), ("lU", "10"), ("IU", "10"),
    ("1O", "10"), ("lO", "10"),
    ("0l", "01"), ("0I", "01"),
    ("l9", "19"), ("I9", "19"), ("1l", "11"),
    ("@", "0"),
    ("2W2", "202"), ("202l", "2021"),
]


def _correct_ocr_typos(date_str: str) -> str:
    """Apply character-level OCR corrections to a string known to be a date.

    Only corrections that are safe in a date context are applied here.
    The substitutions address common digit/letter confusion seen in OCR
    output from scanned bank statements.
    """
    # Remove ordinal suffixes that survived inside slash-separated formats
    date_str = re.sub(
        r"\b(\d{1,2})(?:st|nd|rd|th|uh|ih|\"|h)\b",
        r"\1",
        date_str,
        flags=re.IGNORECASE,
    )

    # Specific single-token replacements
    date_str = date_str.replace("0J", "01")
    date_str = date_str.replace("2O2I", "2021")
    date_str = date_str.replace("l9", "19")
    date_str = date_str.replace("I9", "19")
    date_str = date_str.replace("1l", "11")
    date_str = date_str.replace("?017", "2017")

    # Fix dot separators that should be hyphens: 13.10-2021 → 13-10-2021
    date_str = re.sub(
        r"(\d{1,2})\.(\d{1,2})[.\-](\d{2,4})", r"\1-\2-\3", date_str
    )

    for bad, good in _OCR_DATE_SUBSTITUTIONS:
        date_str = date_str.replace(bad, good)

    return date_str


def normalize_date_strings(rows: List[Row]) -> Tuple[List[Row], int]:
    """Apply date normalization to a list of reconstructed rows.

    Mutates ``row.text`` in-place for any row where normalization changes
    the text.  ``row.original_text`` is preserved from the reconstructor.

    Args:
        rows: Reconstructed row list from transaction_reconstructor.

    Returns:
        A tuple ``(rows, count)`` where *count* is the number of rows
        whose text was modified (including by interpolation).
    """
    normalized_date_count = 0

    for row in rows:
        old_text = row.text
        new_text = old_text

        # Step 1: strip explicit ordinal suffixes before month names
        new_text = ORDINAL_RE.sub(r"\1 ", new_text)

        # Step 2: apply OCR typo corrections inside date-like spans
        def _replace_date_match(match: re.Match) -> str:
            return _correct_ocr_typos(match.group(0))

        new_text = DATE_LIKE_RE.sub(_replace_date_match, new_text)

        # Step 3: fix specific mis-read prefixes
        new_text = re.sub(
            rf"\bIst\s+(?=(?:{MONTHS}))", "1 ", new_text, flags=re.IGNORECASE
        )
        new_text = re.sub(
            rf"\bgh\s*(?=(?:{MONTHS}))", "9 ", new_text, flags=re.IGNORECASE
        )

        if new_text != old_text:
            row.text = new_text
            normalized_date_count += 1

    # Step 4: sequential date interpolation
    interpolated_count = _interpolate_missing_dates(rows)
    normalized_date_count += interpolated_count

    return rows, normalized_date_count


def _parse_date_safely(date_str: str) -> datetime.date | None:
    """Parse a raw date string, returning None on any failure."""
    from dateutil import parser as date_parser

    try:
        clean = re.sub(r"\s+", " ", date_str).strip()
        return date_parser.parse(clean, fuzzy=True).date()
    except Exception:
        return None


def _interpolate_missing_dates(rows: List[Row]) -> int:
    """Prepend an interpolated date to rows that lack one.

    A date is interpolated for row *i* when:
      • Row *i* has no date according to DATE_RE.
      • Row *i-1* and row *i+1* both have parseable dates.
      • The two neighbour dates differ by at most 2 calendar days.

    The format of the interpolated date string mirrors the format
    used by the surrounding rows (numeric vs text months, with or
    without year, matching separator character).

    Args:
        rows: Mutable list of Row objects.

    Returns:
        Number of rows that received an interpolated date prefix.
    """
    count = 0
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

        d1 = _parse_date_safely(prev_date_str)
        d2 = _parse_date_safely(next_date_str)
        if not d1 or not d2:
            continue

        diff = (d2 - d1).days
        interpolated_date: datetime.date | None = None

        if abs(diff) == 1:
            interpolated_date = d1
        elif abs(diff) == 2:
            interpolated_date = d1 + datetime.timedelta(days=1 if diff > 0 else -1)
        elif abs(diff) == 0:
            interpolated_date = d1

        if interpolated_date is None:
            continue

        context = prev_date_str + next_date_str
        has_year = bool(re.search(r"\b\d{4}\b", context))
        has_letters = any(c.isalpha() for c in context)

        if has_letters:
            fmt = "%d %B %Y" if has_year else "%d %B"
            interpolated_str = interpolated_date.strftime(fmt)
        else:
            if has_year:
                interpolated_str = interpolated_date.strftime("%d-%m-%Y")
            else:
                sep = "-"
                if "/" in context:
                    sep = "/"
                elif "." in context:
                    sep = "."
                interpolated_str = interpolated_date.strftime(f"%d{sep}%m")

        old_text = rows[i].text
        rows[i].text = f"{interpolated_str} {old_text}"
        count += 1
        logger.debug(
            "Interpolated date %s for row %d: %r -> %r",
            interpolated_str,
            i,
            old_text,
            rows[i].text,
        )

    return count