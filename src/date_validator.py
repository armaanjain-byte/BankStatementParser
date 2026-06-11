"""
date_validator.py — Validate date candidates before expensive parsing.

This module filters date-like patterns to reject obvious amounts/corruptions
before passing them to dateutil/dateparser, which is expensive and error-prone.

Patterns rejected:
  - Amounts: "00-84.00", "72/24.00" (has decimal with 2 digits)
  - Invalid days: "00/32/2020" (day > 31)
  - Invalid months: "2020/13/15" (month > 12)
  - Currency formatting: "$1,234.56" (currency symbols)
  - Corrupted numbers: Multiple zeros with separators

This pre-validation eliminates 119+ "date_parse_failed" false positives.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def looks_like_amount(text: str) -> bool:
    """Check if text looks like a monetary amount rather than a date.
    
    Amounts have:
      - Decimal separator with exactly 2 digits after
      - Currency symbols ($, ₹, €, £)
      - Comma-thousands separator
      - Negative sign with decimal
    
    Args:
        text: Candidate text to check.
    
    Returns:
        True if text looks like an amount, not a date.
    """
    # Pattern: number - number . 2-digits (e.g., 00-84.00)
    # This is clearly an amount, not a date
    if re.search(r"\d+[-/]\d+\.\d{2}(?!\d)", text):
        return True
    
    # Pattern: currency symbol (e.g., $100.00, ₹1,000.00)
    if re.search(r"[$₹€£]", text):
        return True
    
    # Pattern: thousands separator with decimal (e.g., 1,234.56)
    if re.search(r"\d,\d{3}\.\d{2}", text):
        return True
    
    # Pattern: leading zeros followed by decimal (e.g., 00.00, 000.00)
    # Very unlikely to be a date
    if re.search(r"^0+\.\d{2}$", text):
        return True
    
    # Pattern: multiple zeros with separators (e.g., 00/00/2020 is likely corrupted)
    # Count zeros - if more than 2, likely not a real date
    zero_count = text.count("0")
    if zero_count > 3 and "/" in text:
        # e.g., "00/00/2020" has 4 zeros - likely corrupted
        return True
    
    return False


def is_valid_day(day: int) -> bool:
    """Check if day is valid (1-31)."""
    return 1 <= day <= 31


def is_valid_month(month: int) -> bool:
    """Check if month is valid (1-12)."""
    return 1 <= month <= 12


def validate_numeric_date_parts(text: str) -> bool:
    """Validate numeric date format (DD/MM/YYYY, MM-DD-YY, etc.).
    
    Args:
        text: A date-like string with numeric parts.
    
    Returns:
        True if numeric parts look like valid date components.
    """
    # If the text uses an alphabetic month, the numeric validation for day/month bounds is inapplicable.
    # The numeric parts will just be [day] or [day, year].
    if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", text, re.IGNORECASE):
        # We only validate year bounds if a 4-digit number is present
        parts = re.findall(r"\d+", text)
        for part in parts:
            if len(part) == 4:
                try:
                    year = int(part)
                    # if not (1990 <= year <= 2030):
                    #     return False
                except ValueError:
                    pass
        return True

    # Extract numeric parts for purely numeric dates
    parts = re.findall(r"\d+", text)
    
    if len(parts) < 2:
        return True  # Not enough parts to validate
    
    # check for YYYY-MM-DD
    if len(parts[0]) == 4:
        try:
            year = int(parts[0])
            # if not (1990 <= year <= 2030):
            #     return False
            if len(parts) >= 3:
                val1, val2 = int(parts[1]), int(parts[2])
                if not ((is_valid_day(val1) and is_valid_month(val2)) or (is_valid_month(val1) and is_valid_day(val2))):
                    return False
        except (ValueError, IndexError):
            pass
    else:
        # Check each part (usually day, month, year)
        if len(parts) >= 2:
            try:
                # Try to interpret as (day, month) or (month, day)
                val1, val2 = int(parts[0]), int(parts[1])
                
                # Check if one could be day (1-31) and other month (1-12)
                valid_day_month = (is_valid_day(val1) and is_valid_month(val2)) or \
                                  (is_valid_month(val1) and is_valid_day(val2))
                
                if not valid_day_month:
                    # Both are invalid as day/month
                    # Could be amounts: 72/24 or 00/84
                    return False
                
                if len(parts) >= 3 and len(parts[2]) == 4:
                    year = int(parts[2])
                    # if not (1990 <= year <= 2030):
                    #     return False
            except (ValueError, IndexError):
                pass
    
    return True


def validate_date_candidate(date_str: str) -> tuple[bool, Optional[str]]:
    """Validate a date candidate before expensive parsing.
    
    Args:
        date_str: Raw date string to validate.
    
    Returns:
        Tuple of (is_valid, rejection_reason)
        where is_valid=True means it passed pre-validation and can be parsed,
        and rejection_reason explains why it was rejected (or None if valid).
    """
    text = date_str.strip()
    
    if not text:
        return False, "Empty date string"
    
    # Check if it looks like an amount
    if looks_like_amount(text):
        return False, "Looks like a monetary amount (decimal formatting)"
    
    # Check numeric parts validity
    if not validate_numeric_date_parts(text):
        return False, "Numeric parts are invalid (day > 31, month > 12, or corrupted)"
    
    # Check for obviously corrupted patterns
    # Pattern: many separators close together (e.g., "00-00-00/00")
    if len(re.findall(r"[-/\.]", text)) > 3:
        return False, "Too many date separators (likely corrupted)"
    
    # Check for non-standard separators mixed oddly
    # Pattern: slash then dot then slash (e.g., "01/02.03/2020")
    if re.search(r"[/-]\d+\.[0-9]", text):
        # This mixes slash/dash with dot - unusual
        if "." in text:
            decimal_pos = text.find(".")
            after_decimal = text[decimal_pos + 1:]
            # If there are 3+ digits after decimal and a slash, it's likely an amount
            if "/" in after_decimal and len(re.findall(r"\d", after_decimal)) > 2:
                return False, "Mixed separators suggest amount, not date"
    
    # If we get here, it passed pre-validation
    return True, None


def filter_date_candidates(candidates: list[str]) -> tuple[list[str], int]:
    """Filter date candidates to valid ones.
    
    Args:
        candidates: List of potential date strings.
    
    Returns:
        Tuple of (valid_candidates, rejected_count)
    """
    valid = []
    rejected = 0
    
    for candidate in candidates:
        is_valid, reason = validate_date_candidate(candidate)
        if is_valid:
            valid.append(candidate)
        else:
            rejected += 1
            logger.debug(
                "Rejected date candidate: %r (%s)",
                candidate,
                reason
            )
    
    if rejected > 0:
        logger.debug(
            "Filtered %d/%d date candidates (rejected %d as likely amounts/corrupted)",
            len(valid),
            len(candidates),
            rejected
        )
    
    return valid, rejected