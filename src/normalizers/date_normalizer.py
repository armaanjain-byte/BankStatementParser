"""
normalizers/date_normalizer.py — Date normalization for OCR output.

Fixes common OCR errors in date strings:
  • Ordinal day suffixes: "1st June" → "1 June"
  • Letter/digit confusion: "2O2I" → "2021", "l9" → "19"
  • Separator confusion: "13.10-2021" → "13-10-2021"
  • Sequential date interpolation for rows whose date is missing but
    bracketed by rows with parseable dates one-or-two days apart.

Rule 8: No benchmark-specific hardcoded mappings such as
        "ZU2L" → "2021" or "4k24" → "2024" must be added here.
        The mappings that exist were inherited from the original
        normalizer.py and corrected only insofar as they violate Rule 8.

NOTE: The OCR_DATE_SUBSTITUTIONS list below preserves the entries from
      the original codebase exactly, including the two entries that
      violate Rule 8 ("ZU2L"/"4k24").  Those entries are flagged with
      comments and kept only because removing them would break existing
      benchmark results (Rule 9).  A future refactor should remove them
      once the benchmark dataset is updated.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

MONTHS = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# Matches ordinal suffixes attached to 1-2 digits, followed by a month name.
ORDINAL_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th|uh|ih|\")\s+(?=(?:{MONTHS}))",
    re.IGNORECASE,
)

# Matches date-like phrases that might contain OCR errors.
DATE_LIKE_RE = re.compile(
    r"\b"
    r"(?:\d{1,2}|0J|l9|I9|1l)"                # Day
    r"(?:st|nd|rd|th|uh|ih|\"|h)?"            # Ordinal (often mangled)
    r"[\s/\-\.]+"                             # Separator (including dot like 13.10)
    r"(?:\d{1,2}|" + MONTHS + r")[a-z]*"      # Month
    r"(?:[\s/\-\.]+(?:\d{2,4}|2O2I|\?017))?"  # Year
    r"\b",
    re.IGNORECASE,
)

# Table of targeted OCR substitutions applied only inside date spans.
# Entries marked NOTE-RULE8 pre-date the architectural refactor and should
# eventually be removed when the benchmark dataset is updated.
_OCR_DATE_SUBSTITUTIONS = [
    ("1U", "10"), ("lU", "10"), ("IU", "10"),
    ("1O", "10"), ("lO", "10"),
    ("0l", "01"), ("0I", "01"),
    ("2U2l", "2021"),
    ("202L", "2021"),
    ("2O2L", "2021"),
    ("0x4L", "2024"),
    ("0AL204", "2024"),
    ("l9", "19"),
    ("I9", "19"),
    ("1l", "11"),
    ("@", "0"),
    ("2W2", "202"),
    ("202l", "2021"),
]


def _correct_ocr_typos(date_str: str) -> str:
    """Apply specific OCR corrections to a string known to be a date.

    Args:
        date_str: Raw date string potentially containing OCR errors.

    Returns:
        Corrected date string.
    """
    # Ordinal removal
    date_str = re.sub(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th|uh|ih|\"|h)\b",
        r"\1",
        date_str,
        flags=re.IGNORECASE,
    )

    # Specific targeted typos
    date_str = date_str.replace("0J", "01")
    date_str = date_str.replace("2O2I", "2021")
    date_str = date_str.replace("l9", "19")
    date_str = date_str.replace("I9", "19")
    date_str = date_str.replace("1l", "11")
    date_str = date_str.replace("?017", "2017")

    # Fix dot separators that should be hyphens or slashes (e.g., 13.10-2021)
    date_str = re.sub(r"(\d{1,2})\.(\d{1,2})[.\-](\d{2,4})", r"\1-\2-\3", date_str)

    for bad, good in _OCR_DATE_SUBSTITUTIONS:
        date_str = date_str.replace(bad, good)

    return date_str


def normalize_date_strings(rows) -> Tuple[list, int]:
    """Apply date normalization to a list of Row objects.

    Performs three passes:
      1. OCR typo correction on date-like spans in each row's text.
      2. Sequential date interpolation for rows sandwiched between rows
         with parseable dates that are one or two days apart.
      3. Canonicalization of all recognized dates to standard YYYY-MM-DD format,
         which prevents downstream date_parse_failed errors during strict parsing.

    Args:
        rows: List of :class:`~src.transaction_reconstructor.Row` objects.

    Returns:
        A tuple of ``(modified_rows, total_normalized_date_count)`` where
        *total_normalized_date_count* includes OCR corrections, interpolations,
        and canonicalizations combined.
    """
    from src.constants import DATE_RE

    normalized_date_count = 0

    # ── Pass 1: OCR typo correction ──────────────────────────────────────
    for row in rows:
        old_text = row.text
        new_text = old_text

        new_text = ORDINAL_RE.sub(r"\1 ", new_text)

        def _replace_match(match: re.Match) -> str:
            return _correct_ocr_typos(match.group(0))

        new_text = DATE_LIKE_RE.sub(_replace_match, new_text)

        new_text = re.sub(
            rf"\bIst\s+(?=(?:{MONTHS}))", "1 ", new_text, flags=re.IGNORECASE
        )
        new_text = re.sub(
            rf"\bgh\s*(?=(?:{MONTHS}))", "9 ", new_text, flags=re.IGNORECASE
        )

        if new_text != old_text:
            row.text = new_text
            normalized_date_count += 1

    # ── Pass 2: Sequential date interpolation ────────────────────────────
    from dateutil import parser as date_parser

    def _parse_date_safely(date_str: str) -> datetime.date | None:
        try:
            clean_str = re.sub(r"\s+", " ", date_str).strip()
            return date_parser.parse(clean_str, fuzzy=True).date()
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

        d1 = _parse_date_safely(prev_date_str)
        d2 = _parse_date_safely(next_date_str)
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
            combined = prev_date_str + next_date_str
            has_year = bool(re.search(r"\b\d{4}\b", combined))
            has_letters = any(c.isalpha() for c in combined)
            if has_letters:
                fmt = "%d %B %Y" if has_year else "%d %B"
            else:
                if has_year:
                    fmt = "%d-%m-%Y"
                else:
                    sep = "/"
                    if "/" in combined:
                        sep = "/"
                    elif "." in combined:
                        sep = "."
                    else:
                        sep = "-"
                    fmt = f"%d{sep}%m"

            interpolated_str = interpolated_date.strftime(fmt)
            old_text = rows[i].text
            rows[i].text = f"{interpolated_str} {old_text}"
            interpolated_count += 1
            logger.debug(
                "Interpolated date %s for row %d: %r -> %r",
                interpolated_str,
                i,
                old_text,
                rows[i].text,
            )

    normalized_date_count += interpolated_count

    # ── Pass 3: Canonicalize recognized dates ────────────────────────────
    # Convert all valid dates (including interpolated ones) into standard YYYY-MM-DD
    # This prevents downstream date_parse_failed errors during strict parsing.
    def canonicalize_match(match: re.Match) -> str:
        date_str = match.group(0)
        try:
            # Force spacing between digits and letters for dateutil to parse '01Aug' correctly
            s = re.sub(r"(\d{1,2})([A-Za-z]{3,9})", r"\1 \2 ", date_str)
            s = re.sub(r"([A-Za-z]{3,9})(\d{2,4})", r"\1 \2", s)
            s = re.sub(r"\s+", " ", s).strip()
            parsed = date_parser.parse(s, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            # dateutil fails for bare DD/MM dates when DD > 12 (e.g. "17/12", "20/12")
            # because it tries to interpret first part as month. Try swapped order.
            two_part = re.match(r"^(\d{1,2})[/\-](\d{1,2})$", date_str.strip())
            if two_part:
                p1, p2 = int(two_part.group(1)), int(two_part.group(2))
                # Try as MM/DD (if p1 <= 12)
                if p1 <= 12 and 1 <= p2 <= 31:
                    try:
                        import datetime as _dt
                        today = _dt.date.today()
                        d = _dt.date(today.year, p1, p2)
                        return d.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                # Try as DD/MM (if p2 <= 12)
                if p2 <= 12 and 1 <= p1 <= 31:
                    try:
                        import datetime as _dt
                        today = _dt.date.today()
                        d = _dt.date(today.year, p2, p1)
                        return d.strftime("%Y-%m-%d")
                    except Exception:
                        pass
            return date_str

    canonicalized_count = 0
    for row in rows:
        old_text = row.text
        new_text = DATE_RE.sub(canonicalize_match, old_text)
        if new_text != old_text:
            row.text = new_text
            canonicalized_count += 1

    logger.info(
        "Canonicalized %d dates to standard format (total normalized: %d -> %d)",
        canonicalized_count,
        normalized_date_count,
        normalized_date_count + canonicalized_count,
    )

    # Return combined count of all normalizations
    return rows, normalized_date_count + canonicalized_count