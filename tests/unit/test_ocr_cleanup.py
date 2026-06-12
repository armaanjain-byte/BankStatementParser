"""
Unit tests for src.normalizers.ocr_cleanup.

Pipeline stage: generic OCR text cleanup (used ad-hoc; not currently wired
into the main pipeline per main.py, but exported and presumably intended
for future token-level cleanup).

Functions under test:
  collapse_whitespace(text) -> str
  strip_noise_chars(text) -> str
  normalize_unicode(text) -> str
  clean_token_text(text) -> str  (= normalize_unicode + collapse_whitespace)
"""

from __future__ import annotations

import pytest

from src.normalizers.ocr_cleanup import (
    clean_token_text,
    collapse_whitespace,
    normalize_unicode,
    strip_noise_chars,
)


@pytest.mark.unit
class TestCollapseWhitespace:
    def test_multiple_spaces_collapsed(self):
        """WHY: OCR column alignment often inserts multiple spaces between
        tokens; downstream regexes (DATE_RE, MONEY_RE) expect normal
        single-space-separated text."""
        assert collapse_whitespace("a    b") == "a b"

    def test_leading_trailing_whitespace_stripped(self):
        assert collapse_whitespace("  a b  ") == "a b"

    def test_tabs_and_nbsp_collapsed(self):
        """WHY: non-breaking spaces (\\xa0) and tabs are common OCR
        artifacts and must collapse like regular spaces."""
        assert collapse_whitespace("a\t\xa0 b") == "a b"


@pytest.mark.unit
class TestStripNoiseChars:
    def test_leading_pipe_removed(self):
        """WHY: table-border characters like '|' frequently get attached to
        the first/last token of a row."""
        assert strip_noise_chars("|Description") == "Description"

    def test_trailing_punctuation_removed(self):
        assert strip_noise_chars("Total.") == "Total"

    def test_internal_decimal_point_preserved(self):
        """WHY: strip_noise_chars must NOT eat the decimal point inside
        '1,234.56' -- only leading/trailing noise.
        BUG PREVENTED: '1,234.56' becoming '1,234' or '1,23456'."""
        assert strip_noise_chars("1,234.56") == "1,234.56"

    def test_pure_separator_row_becomes_empty(self):
        """WHY: a row that is purely '---------' or similar visual separator
        should strip down to an empty string, allowing row_classifier to
        identify it as a separator."""
        assert strip_noise_chars("----------") == ""


@pytest.mark.unit
class TestNormalizeUnicode:
    def test_curly_quotes_replaced(self):
        assert normalize_unicode("\u2018hello\u2019") == "'hello'"
        assert normalize_unicode("\u201chello\u201d") == '"hello"'

    def test_em_dash_and_en_dash_replaced_with_hyphen(self):
        """WHY: dates sometimes use em/en-dashes ('12\u201312\u20132021')
        instead of hyphens; DATE_RE expects '-' or '/'.
        BUG PREVENTED: a valid-looking date failing DATE_RE purely because
        of a Unicode dash character."""
        assert normalize_unicode("12\u201312\u20132021") == "12-12-2021"
        assert normalize_unicode("12\u201412\u20142021") == "12-12-2021"

    def test_nbsp_replaced_with_space(self):
        assert normalize_unicode("a\u00a0b") == "a b"


@pytest.mark.unit
class TestCleanTokenText:
    def test_pipeline_applies_unicode_then_whitespace(self):
        """WHY: clean_token_text = normalize_unicode -> collapse_whitespace.
        This test pins the ORDER of operations: unicode normalization must
        happen first so that, e.g., a converted nbsp gets collapsed by the
        whitespace pass afterward."""
        result = clean_token_text("\u201cHello\u00a0\u00a0World\u201d")
        assert result == '"Hello World"'

    def test_strip_noise_chars_not_included_by_design(self):
        """WHY: the module docstring explicitly states strip_noise_chars is
        deliberately NOT part of clean_token_text because it can destroy
        valid characters in date separators and amount tokens. This test
        documents that design decision so a future "helpful" refactor that
        adds it back doesn't silently change behavior for every token."""
        result = clean_token_text("|13-10-2021|")
        assert result == "|13-10-2021|"  # noise chars NOT stripped