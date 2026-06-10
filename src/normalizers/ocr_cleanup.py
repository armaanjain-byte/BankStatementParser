"""
normalizers/ocr_cleanup.py — Generic OCR character-level cleanup utilities.

This module contains character substitution tables and helper functions
that are shared between the date and amount normalizers.  It does NOT
contain domain-specific normalization logic (date parsing, amount
validation, etc.) — those live in date_normalizer.py and amount_normalizer.py
respectively.

Exposed for direct use by other modules that need low-level OCR correction
without pulling in the full normalizer stack.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Common digit/letter confusion characters seen in scanned bank statements.
# These are safe to substitute in any numeric context.
# ---------------------------------------------------------------------------

#: Characters OCR frequently confuses for digits.
#: Maps mis-read character → correct digit.
DIGIT_SUBSTITUTIONS: dict[str, str] = {
    "O": "0",
    "o": "0",
    "C": "0",
    "c": "0",
    "I": "1",
    "l": "1",
}

#: Currency / noise characters that should be stripped from the start
#: of an amount string.
LEADING_NOISE_CHARS: frozenset[str] = frozenset("~$₹€£")


def strip_leading_noise(text: str) -> str:
    """Remove leading currency symbols and noise characters from *text*.

    Args:
        text: Raw string that may begin with currency/noise characters.

    Returns:
        The string with leading noise removed and surrounding whitespace
        stripped.
    """
    s = text.strip()
    while s and s[0] in LEADING_NOISE_CHARS:
        s = s[1:]
    return s.strip()


def apply_digit_substitutions(text: str, substitutions: dict[str, str] | None = None) -> str:
    """Replace mis-read characters with their correct digit equivalents.

    Substitutions are only applied when the candidate character is
    flanked by digits or decimal/thousands separators, reducing false
    positives on alphabetic text.

    Args:
        text:          Input string containing potential digit mis-reads.
        substitutions: Mapping of bad_char → correct_char.  Defaults to
                       :data:`DIGIT_SUBSTITUTIONS`.

    Returns:
        The corrected string.
    """
    sub_map = substitutions if substitutions is not None else DIGIT_SUBSTITUTIONS
    result = []
    for i, ch in enumerate(text):
        if ch in sub_map:
            prev_ok = i > 0 and (text[i - 1].isdigit() or text[i - 1] in ".,")
            next_ok = i < len(text) - 1 and (text[i + 1].isdigit() or text[i + 1] in ".,")
            result.append(sub_map[ch] if (prev_ok or next_ok) else ch)
        else:
            result.append(ch)
    return "".join(result)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends.

    Args:
        text: Input string.

    Returns:
        String with normalised internal whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()