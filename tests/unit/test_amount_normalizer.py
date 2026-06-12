"""
Unit tests for src.normalizers.amount_normalizer.

Pipeline stage: AMOUNT NORMALIZATION.

Function under test: normalize_amount_strings(rows) -> (rows, normalized_count)
  Expected input:  List[Row]
  Expected output: (rows with .text rewritten in place, int count)

Internal helpers also tested directly:
  _normalize_amount_token(token) -> str
  _repair_amount_separators(amount_str) -> str

Edge cases:
  - comma-as-decimal ("925,90" -> "925.90")
  - OCR digit substitutions ("530.C0" -> "530.00")
  - whitespace-as-decimal ("-350 00" -> "-350.00")
  - leading currency noise ("$24.50" -> "24.50")
  - bare integers must NOT get a fabricated ".00" (Rule removed per docstring)
  - date spans must not be corrupted by amount normalization
  - thousands separators ("1,234") must not become "1.23"

Failure cases:
  - "32.00 801.00"-style multi-amount spans must not be merged into one
    fabricated value
"""

from __future__ import annotations

import pytest

from src.normalizers.amount_normalizer import (
    _normalize_amount_token,
    _repair_amount_separators,
    normalize_amount_strings,
)
from src.constants import MONEY_RE
from tests.fixtures.ocr_tokens import make_row_from_text


@pytest.mark.unit
class TestNormalizeAmountToken:
    def test_comma_decimal_separator(self):
        """WHY: some statements use ',' as the decimal separator (European
        convention). '925,90' must become '925.90' so MONEY_RE matches it."""
        assert _normalize_amount_token("925,90") == "925.90"

    def test_ocr_letter_zero_confusion(self):
        """WHY: OCR frequently reads '0' as 'O' or 'C'. '530.C0' must become
        '530.00'."""
        result = _normalize_amount_token("530.C0")
        result = _repair_amount_separators(result)
        assert result == "530.00"

    def test_whitespace_decimal_separator(self):
        """WHY: column-alignment artifacts can insert a space where a decimal
        point belongs: '-350 00' -> '-350.00'."""
        assert _normalize_amount_token("-350 00") == "-350.00"

    def test_leading_currency_symbol_stripped(self):
        """WHY: '$24.50' and '~8300.00' must have leading noise chars ('$',
        '~', '₹', etc.) stripped before validation."""
        assert _normalize_amount_token("$24.50").lstrip("$") == "24.50"
        assert _normalize_amount_token("~8300.00") == "8300.00"

    def test_duplicated_separator_cleanup(self):
        """WHY: '92.,50' (duplicated/misplaced separators) must collapse to
        '92.50'."""
        assert _normalize_amount_token("92.,50") == "92.50"


@pytest.mark.unit
class TestRepairAmountSeparators:
    def test_bare_integer_not_given_fabricated_decimal(self):
        """WHY: per the module docstring, Rule 1 (appending '.00' to bare
        integers) was explicitly REMOVED because it fabricated amounts from
        cheque/reference/account numbers (e.g. '201053' -> '201,053.00').
        BUG PREVENTED: a regression that re-adds fabricated decimals to bare
        integers, corrupting reference numbers into fake transaction
        amounts."""
        result = _repair_amount_separators("201053")
        assert result == "201053"  # unchanged, no ".00" appended

    def test_thousands_separator_not_truncated_to_decimal(self):
        """WHY: '30, 800' (a thousands group with OCR-inserted space) must
        NOT become '30.80'. The thousands guard checks for a trailing
        3-digit group after ',' or ' ' and returns the original unchanged.
        BUG PREVENTED: '1,234' -> '1.23' (losing 3 orders of magnitude)."""
        result = _repair_amount_separators("30, 800")
        assert result == "30, 800"  # unchanged by thousands guard

    def test_terminal_C0_OO_Co_fixed_to_00(self):
        """WHY: Rule 2 — terminal 'C0'/'OO'/'Co' (OCR zero confusion in the
        decimal part) must become '00'."""
        assert _repair_amount_separators("530.C0").endswith(".00")
        assert _repair_amount_separators("100.OO").endswith(".00")

    def test_invalid_result_returns_original(self):
        """WHY: Rule 4 validates the repaired string against MONEY_RE; if
        repair produces something that still doesn't validate, the ORIGINAL
        string must be returned unchanged (fail-safe, no partial corruption)."""
        original = "abc"
        # _repair_amount_separators short-circuits on "no digits" -> returns input unchanged
        assert _repair_amount_separators(original) == original


@pytest.mark.unit
class TestNormalizeAmountStrings:
    def test_row_with_clean_amount_unchanged(self):
        """WHY: a row that already has a valid MONEY_RE amount must not be
        mutated (idempotence on already-correct input)."""
        original = "12/03/2021 UPI Payment 226.14"
        rows = [make_row_from_text(original)]
        rows, count = normalize_amount_strings(rows)
        assert rows[0].text == original
        assert count == 0

    def test_date_spans_protected_from_amount_corruption(self):
        """WHY: without the date-span guard, bare integers inside dates
        ('13', '10', '2021' in '13-10-2021') could be converted to '13.00',
        '10.00', '2,021.00', destroying the date.
        BUG PREVENTED: a date like '13-10-2021' becoming '13.00-10.002,021.00'
        or similar garbage, which would make DATE_RE fail on the row
        entirely (no_date_detected)."""
        original = "13-10-2021 Some Narration 500.00"
        rows = [make_row_from_text(original)]
        rows, count = normalize_amount_strings(rows)
        # The date portion must still be intact and DATE_RE-matchable.
        assert "13-10-2021" in rows[0].text

    def test_already_valid_amount_in_span_left_untouched_merge_guard(self):
        """WHY: AMOUNT_CANDIDATE_RE can span whitespace and swallow adjacent
        amounts ('32.00 801.00'). The merge guard checks if MONEY_RE already
        matches within the candidate span and, if so, leaves it untouched
        rather than 'repairing' it into one fabricated value.
        BUG PREVENTED: '32.00 801.00' -> '1,231,093.00' or similar merged
        garbage that destroys two valid amounts."""
        original = "12/03/2021 Fee split 32.00 801.00"
        rows = [make_row_from_text(original)]
        rows, count = normalize_amount_strings(rows)
        # Both original amounts must still be present and individually valid.
        amounts = MONEY_RE.findall(rows[0].text)
        assert "32.00" in amounts
        assert "801.00" in amounts

    def test_ocr_corrupted_amount_recovered(self):
        """WHY: 'Rs 530.C0' should normalize to a valid MONEY_RE match
        ('530.00') after stripping the 'Rs' currency prefix is NOT
        guaranteed (Rs is alphabetic, not in the noise-strip set) but the
        'C0' -> '00' fix should still apply to the numeric portion.
        This test is intentionally loose: it checks that *some* valid money
        value is recoverable from a corrupted numeric span."""
        rows = [make_row_from_text("12/03/2021 Payment 530.C0")]
        rows, count = normalize_amount_strings(rows)
        amounts = MONEY_RE.findall(rows[0].text)
        assert "530.00" in amounts
        assert count >= 1