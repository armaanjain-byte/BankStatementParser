"""
Unit tests for src.date_validator.

Pipeline stage: DATE VALIDATION (pre-filter before expensive dateutil parsing).

Function under test: validate_date_candidate(date_str) -> (is_valid, reason)
  Expected input:  a DATE_RE match string
  Expected output: (True, None) if it looks like a genuine date,
                    (False, "<reason>") if it looks like an amount or is
                    structurally invalid.

REGRESSION ANCHORS: the 4 'date_parse_failed' entries in
out/failure_reports.json are "1/13.00", "00-6.00", "00-7.00", "00-7.00".
This module's docstring claims it eliminates "119+ date_parse_failed false
positives" by rejecting amount-shaped strings before dateutil ever sees
them. These tests confirm those exact strings ARE rejected (correctly,
since they're amounts) -- i.e. the 4 remaining date_parse_failed entries in
the report are appropriately-rejected amounts being miscounted as date
failures by evaluation.py's classification, not actual date_validator bugs.
"""

from __future__ import annotations

import pytest

from src.date_validator import (
    filter_date_candidates,
    is_valid_day,
    is_valid_month,
    looks_like_amount,
    validate_date_candidate,
    validate_numeric_date_parts,
)


@pytest.mark.unit
class TestLooksLikeAmount:
    @pytest.mark.parametrize("text", ["1/13.00", "00-6.00", "00-7.00", "$1,234.56", "00/00/2020"])
    def test_amount_shaped_strings_detected(self, text):
        """WHY: these exact strings are the 'candidate' values for the 4
        date_parse_failed records in out/failure_reports.json. They are
        amounts (or amount-like corruption), not dates, and must be flagged
        as such."""
        assert looks_like_amount(text) is True

    @pytest.mark.parametrize("text", ["13-10-2021", "01 June 2021", "23/12/2019"])
    def test_genuine_dates_not_flagged_as_amounts(self, text):
        assert looks_like_amount(text) is False


@pytest.mark.unit
class TestValidDayMonth:
    def test_valid_day_range(self):
        assert is_valid_day(1) is True
        assert is_valid_day(31) is True
        assert is_valid_day(0) is False
        assert is_valid_day(32) is False

    def test_valid_month_range(self):
        assert is_valid_month(1) is True
        assert is_valid_month(12) is True
        assert is_valid_month(0) is False
        assert is_valid_month(13) is False


@pytest.mark.unit
class TestValidateNumericDateParts:
    def test_day_month_year_order_accepted(self):
        """WHY: 'DD/MM/YYYY' (e.g. 23/12/2019) — 23 is a valid day, 12 a
        valid month."""
        assert validate_numeric_date_parts("23/12/2019") is True

    def test_month_day_year_order_accepted(self):
        """WHY: US-style 'MM/DD/YYYY' (e.g. 12/23/2019) must also pass —
        the function accepts either ordering as long as one interpretation
        is valid."""
        assert validate_numeric_date_parts("12/23/2019") is True

    def test_invalid_month_and_day_rejected(self):
        """WHY: '2020/13/15' has year=2020, then 13 and 15 -- neither
        (13=month invalid, 15=day) nor (15=month invalid, 13=day) is a
        valid day/month pair. Must be rejected."""
        assert validate_numeric_date_parts("2020/13/15") is False

    def test_alphabetic_month_skips_numeric_bounds_check(self):
        """WHY: '01 June 2021' has an alphabetic month; numeric day/month
        bounds checking is inapplicable (only the day '01' and year '2021'
        are numeric), so this must return True without attempting
        day/month validation on those two numbers."""
        assert validate_numeric_date_parts("01 June 2021") is True


@pytest.mark.unit
class TestValidateDateCandidate:
    def test_empty_string_rejected(self):
        is_valid, reason = validate_date_candidate("")
        assert is_valid is False
        assert reason == "Empty date string"

    @pytest.mark.parametrize(
        "candidate",
        ["1/13.00", "00-6.00", "00-7.00"],
        ids=["1_13.00", "00-6.00", "00-7.00"],
    )
    def test_failure_report_date_parse_failed_samples_rejected(self, candidate):
        """REGRESSION: these are the literal 'candidate' strings from the
        date_parse_failed entries in out/failure_reports.json (statements
        25 and 26). They are correctly rejected here as amount-shaped, not
        genuine dates. If this test starts failing (i.e. one of these
        becomes 'valid'), a future change has weakened looks_like_amount()
        and may reintroduce false-positive date matches on amount columns."""
        is_valid, reason = validate_date_candidate(candidate)
        assert is_valid is False
        assert reason == "Looks like a monetary amount (decimal formatting)"

    def test_genuine_date_accepted(self):
        is_valid, reason = validate_date_candidate("13-10-2021")
        assert is_valid is True
        assert reason is None

    def test_too_many_separators_rejected(self):
        """WHY: '00-00-00/00' has 4 separators (> 3), flagged as 'likely
        corrupted' regardless of looks_like_amount/numeric-parts checks."""
        is_valid, reason = validate_date_candidate("00-00-00/00")
        assert is_valid is False


@pytest.mark.unit
class TestFilterDateCandidates:
    def test_mixed_candidates_split_correctly(self):
        """WHY: filter_date_candidates must partition a list into
        (valid_dates, rejected_count) using validate_date_candidate per
        item."""
        candidates = ["13-10-2021", "1/13.00", "23/12/2019", "00-6.00"]
        valid, rejected = filter_date_candidates(candidates)
        assert valid == ["13-10-2021", "23/12/2019"]
        assert rejected == 2

    def test_empty_input_returns_empty_with_zero_rejections(self):
        valid, rejected = filter_date_candidates([])
        assert valid == []
        assert rejected == 0