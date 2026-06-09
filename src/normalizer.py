"""
normalizer.py - Dedicated normalization layer for date parsing robustness.

Fixes common OCR errors in dates and normalizes ordinal day formats
without mutating text globally.
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
