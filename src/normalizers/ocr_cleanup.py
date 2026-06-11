"""
normalizers/ocr_cleanup.py — Generic OCR token-text cleanup utilities.

Contains transformations that apply to arbitrary OCR text regardless of
whether it represents a date, amount, or description.

Currently exposes:
  • clean_token_text()   — normalise whitespace and strip invisible chars
  • strip_noise_chars()  — remove leading/trailing punctuation noise
  • collapse_whitespace() — collapse multi-space runs to single spaces

These helpers are intentionally narrow: they do not know about dates or
amounts.  Domain-specific fixes belong in date_normalizer.py or
amount_normalizer.py respectively.
"""

from __future__ import annotations

import re
import unicodedata


# Characters that are commonly introduced by OCR and carry no semantic value.
_NOISE_CHARS_RE = re.compile(r"^[\s\-–—_|/\\.,;:!?@#$%^&*()[\]{}<>\"'`~]+|"
                              r"[\s\-–—_|/\\.,;:!?@#$%^&*()[\]{}<>\"'`~]+$")

# Any run of whitespace (spaces, tabs, non-breaking spaces, etc.)
_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace characters to a single ASCII space.

    Args:
        text: Input string, possibly containing multiple spaces, tabs, or
              non-breaking spaces introduced by OCR column alignment.

    Returns:
        String with all internal whitespace runs collapsed to a single space,
        and leading/trailing whitespace removed.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_noise_chars(text: str) -> str:
    """Remove leading and trailing punctuation/noise characters.

    OCR engines sometimes attach stray punctuation to tokens (e.g. a
    leading pipe ``|`` from a table border, or a trailing full-stop that
    belongs to the sentence rather than the token).  This function strips
    those edge characters while preserving internal punctuation that is
    semantically significant (e.g. the decimal point in "1,234.56").

    Args:
        text: Input OCR token text.

    Returns:
        Text with noise characters removed from both ends.
    """
    return _NOISE_CHARS_RE.sub("", text)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to their closest ASCII equivalents.

    Replaces common OCR-introduced Unicode lookalikes (e.g. curly quotes,
    em-dashes, non-breaking spaces) with standard ASCII characters.

    Args:
        text: Input string potentially containing Unicode lookalikes.

    Returns:
        NFC-normalized string with common lookalikes replaced.
    """
    # NFC normalization first
    text = unicodedata.normalize("NFC", text)

    # Replace typographic quotes with straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Replace en-dash / em-dash with hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ")

    return text


def clean_token_text(text: str) -> str:
    """Apply the full generic cleanup pipeline to a single OCR token.

    Pipeline:
      1. :func:`normalize_unicode`
      2. :func:`collapse_whitespace`

    Note: :func:`strip_noise_chars` is deliberately *not* included here
    because it can destroy valid characters in date separators and amount
    tokens.  Callers that need aggressive edge-stripping should call it
    explicitly.

    Args:
        text: Raw OCR token text.

    Returns:
        Cleaned token text.
    """
    text = normalize_unicode(text)
    text = collapse_whitespace(text)
    return text