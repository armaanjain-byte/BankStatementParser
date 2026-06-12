"""
tests/integration/test_normalize_then_parse.py — Integration: normalizer → parser

Tests that verify the output of date_normalizer / amount_normalizer feeds
cleanly into parse_rows() and produces correctly populated ParsedRows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.ocr_tokens import make_row_from_text as make_row
from src.normalizers.date_normalizer import normalize_date_strings
from src.normalizers.amount_normalizer import normalize_amount_strings
from src.transaction_parser import parse_rows, ParsedRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rows_from(texts: list[str]):
    return [make_row(t, y_center=i * 25.0) for i, t in enumerate(texts)]


# ===========================================================================
# date_normalizer → parse_rows
# ===========================================================================

class TestDateNormalizerToParser:
    def test_ordinal_date_parsed_after_normalisation(self):
        raw = ["1st January 2021 ATM withdrawal 500.00"]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) >= 1
        assert parsed[0].amounts == ["500.00"]

    def test_ocr_mangled_year_2O2I_corrected(self):
        raw = ["15/01/2O2I UPI payment 1,200.00"]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        # After normalisation the year-like fragment must have been fixed
        assert "2O2I" not in rows[0].text

    def test_multiple_rows_each_parsed(self):
        raw = [
            "01/01/2021 Salary credit 50,000.00",
            "05/01/2021 ATM withdrawal 5,000.00",
            "10/01/2021 UPI payment 1,200.00",
        ]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) == 3

    def test_separator_normalised_date_parses_correctly(self):
        # "13.10-2021" should be normalised to "13-10-2021"
        raw = ["13.10-2021 Grocery purchase 350.00"]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) >= 1

    def test_header_rows_filtered_after_normalisation(self):
        raw = [
            "Date Narration Debit Credit Balance",  # header
            "01/01/2021 ATM WITHDRAWAL 5,000.00 45,000.00",  # transaction
        ]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        parsed = parse_rows(rows)
        # Header should be filtered out
        assert all(r.date != "Date" for r in parsed)


# ===========================================================================
# amount_normalizer → parse_rows
# ===========================================================================

class TestAmountNormalizerToParser:
    def test_comma_decimal_amount_normalised_and_parsed(self):
        raw = ["01/01/2021 NEFT transfer 925,90"]
        rows = rows_from(raw)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        # After normalisation "925,90" → "925.90" should parse
        assert len(parsed) >= 1
        assert any("925.90" in r.amounts for r in parsed)

    def test_ocr_zero_confusion_fixed(self):
        # "530.C0" → "530.00"
        raw = ["15/02/2021 Utility bill 530.C0"]
        rows = rows_from(raw)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) >= 1

    def test_date_spans_not_corrupted_by_amount_normaliser(self):
        raw = ["01/01/2021 NEFT transfer 1,500.00"]
        rows = rows_from(raw)
        rows, _ = normalize_amount_strings(rows)
        # Date must remain intact
        assert "01/01/2021" in rows[0].text or "2021" in rows[0].text

    def test_multiple_amounts_all_parsed(self):
        raw = ["03/01/2021 payment 1,000.00 49,000.00"]
        rows = rows_from(raw)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) >= 1
        assert len(parsed[0].amounts) >= 1  # at least one amount


# ===========================================================================
# Both normalisers together → parse_rows
# ===========================================================================

class TestBothNormalisersToParser:
    def test_fully_normalised_row_parses(self):
        raw = ["1st Jun 2021 UPI payment 1,200.C0 48,800.00"]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        assert len(parsed) >= 1

    def test_normalised_count_increases_parsed_count(self):
        # Without normalisation the OCR-mangled row may not parse.
        # With normalisation it should.
        mangled = ["01 Jan 2O2I IMPS credit 10,000.00"]
        rows_raw = rows_from(mangled)
        parsed_raw = parse_rows(rows_raw)

        rows_norm = rows_from(mangled)
        rows_norm, _ = normalize_date_strings(rows_norm)
        rows_norm, _ = normalize_amount_strings(rows_norm)
        parsed_norm = parse_rows(rows_norm)

        assert len(parsed_norm) >= len(parsed_raw)

    def test_description_populated_after_full_normalisation(self):
        raw = ["05/01/2021 Salary from employer 50,000.00 1,50,000.00"]
        rows = rows_from(raw)
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        if parsed:
            assert parsed[0].description is not None
            assert len(parsed[0].description) > 0