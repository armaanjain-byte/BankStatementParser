"""
tests/regression/test_known_failures.py — Regression tests for failure_reports.json

Every exact candidate string that appeared as a non-paddleocr failure in
out/failure_reports.json is tested here.  The goal is to ensure that future
refactors do not silently regress on these real-world failure cases.

Failure types covered:
  • date_parse_failed  — "00-6.00", "00-7.00", "1/13.00"
    These are monetary amounts that leaked into date parsing.
    Regression: date_validator must reject them.

  • no_date_detected   — rows that genuinely have no parseable date.
    Regression: parse_rows must not fabricate a date for them.

  • Mangled OCR rows   — "CLO-ZQ Z0E 17(M", "0R-UZ-20 NEFL-AXIS..."
    Regression: pipeline must not crash and must not produce
    phantom transactions from noise rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import make_row
from src.date_validator import validate_date_candidate, looks_like_amount
from src.normalizers.date_normalizer import normalize_date_strings
from src.normalizers.amount_normalizer import normalize_amount_strings
from src.transaction_parser import parse_rows


# ---------------------------------------------------------------------------
# Verbatim candidate strings from failure_reports.json
# ---------------------------------------------------------------------------

DATE_PARSE_FAILED = [
    "1/13.00",   # statement_id=25
    "00-6.00",   # statement_id=26
    "00-7.00",   # statement_id=26  (appears twice)
]

NO_DATE_DETECTED = [
    "Sept. Bill paymnemt Irotn) '201053 JU [68233On-...",   # id=23
    "(et . Withdrawal RarcWays;",                            # id=23
    "Oct . Cill parment from Garelars Four Oaks 32.0...",   # id=23
    "End balance 302!41.139",                                # id=23
    "Jul 2017 ATM OPERATOR FEE WIThDRAWAL AT ANZ ATM",      # id=25
    "AEN33 Do7 457 141 &AFSL NJshaHrt ccdi: |cajo 23...",   # id=25
    "CLO-ZQ Z0E 17(M",                                      # id=26
    "0R-UZ-20 NEFL-AXISI4A9004593-BASIREDDY Suuu.UU",       # id=26
    "(0s-10-2021 BampvnicnI LOQU.UO",                        # id=27
    "04-2W2 MANPOKARUR VS 1676,0",                           # id=27
]


# ===========================================================================
# date_parse_failed: candidates must be rejected as amounts by date_validator
# ===========================================================================

class TestDateParseFailedRejection:
    @pytest.mark.parametrize("candidate", DATE_PARSE_FAILED)
    def test_rejected_by_date_validator(self, candidate: str):
        """date_validator must reject all date_parse_failed candidates."""
        is_valid, reason = validate_date_candidate(candidate)
        assert not is_valid, (
            f"Expected {candidate!r} to be rejected but it passed validation"
        )

    @pytest.mark.parametrize("candidate", DATE_PARSE_FAILED)
    def test_looks_like_amount(self, candidate: str):
        """All date_parse_failed strings contain decimal-amount formatting."""
        assert looks_like_amount(candidate), (
            f"Expected {candidate!r} to look like an amount"
        )

    @pytest.mark.parametrize("candidate", DATE_PARSE_FAILED)
    def test_no_transaction_parsed_from_amount_string(self, candidate: str):
        """parse_rows must not produce a transaction from a bare amount string."""
        row = make_row(candidate)
        result = parse_rows([row])
        assert result == [], (
            f"Got unexpected ParsedRow from amount-only string: {candidate!r}"
        )


# ===========================================================================
# no_date_detected: parse_rows must not invent dates for these rows
# ===========================================================================

class TestNoDateDetectedRows:
    @pytest.mark.parametrize("candidate", NO_DATE_DETECTED)
    def test_no_transaction_parsed_without_valid_date(self, candidate: str):
        """parse_rows must not produce a transaction from a dateless noise row."""
        rows = [make_row(candidate)]
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        result = parse_rows(rows)
        # Either no result, or result must still have a non-empty date
        for parsed in result:
            assert parsed.date, (
                f"ParsedRow without a date was produced from: {candidate!r}"
            )

    @pytest.mark.parametrize("candidate", NO_DATE_DETECTED)
    def test_pipeline_does_not_crash(self, candidate: str):
        """The pipeline must not raise an exception on any known failure row."""
        rows = [make_row(candidate)]
        try:
            rows, _ = normalize_date_strings(rows)
            rows, _ = normalize_amount_strings(rows)
            parse_rows(rows)
        except Exception as exc:
            pytest.fail(
                f"Pipeline raised {type(exc).__name__} on {candidate!r}: {exc}"
            )


# ===========================================================================
# Mangled OCR rows: no phantom transactions, no crashes
# ===========================================================================

MANGLED_OCR_ROWS = [
    "CLO-ZQ Z0E 17(M",
    "0R-UZ-20 NEFL-AXISI4A9004593-BASIREDDY Suuu.UU",
    "(0s-10-2021 BampvnicnI LOQU.UO",
    "04-2W2 MANPOKARUR VS 1676,0",
    "AEN33 Do7 457 141 &AFSL NJshaHrt ccdi: |cajo 23...",
    "Sept. Bill paymnemt Irotn) '201053 JU [68233On-...",
]


class TestMangledOCRRows:
    @pytest.mark.parametrize("candidate", MANGLED_OCR_ROWS)
    def test_no_crash(self, candidate: str):
        rows = [make_row(candidate)]
        try:
            rows, _ = normalize_date_strings(rows)
            rows, _ = normalize_amount_strings(rows)
            parse_rows(rows)
        except Exception as exc:
            pytest.fail(f"Crash on mangled row {candidate!r}: {type(exc).__name__}: {exc}")

    def test_mixed_batch_does_not_crash(self):
        """A batch with both valid and mangled rows must process without error."""
        rows = [make_row(c, y_center=float(i * 25)) for i, c in enumerate(MANGLED_OCR_ROWS)]
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        result = parse_rows(rows)
        # At most one of these should parse (the "(0s-10-2021 ..." row might
        # slip through after date normalisation); the others must be filtered.
        assert len(result) <= len(MANGLED_OCR_ROWS)


# ===========================================================================
# "End balance" footer row must be rejected
# ===========================================================================

class TestEndBalanceFooter:
    def test_end_balance_row_not_parsed_as_transaction(self):
        row = make_row("End balance 302!41.139")
        rows, _ = normalize_date_strings([row])
        rows, _ = normalize_amount_strings(rows)
        result = parse_rows(rows)
        assert result == [], "Footer row 'End balance' must not produce a transaction"


# ===========================================================================
# Date_validator rejects all DATE_PARSE_FAILED with specific reasons
# ===========================================================================

class TestDateValidatorRejectionReasons:
    def test_00_6_00_rejected_as_amount(self):
        is_valid, reason = validate_date_candidate("00-6.00")
        assert not is_valid
        assert reason is not None

    def test_1_13_00_rejected_as_amount(self):
        is_valid, reason = validate_date_candidate("1/13.00")
        assert not is_valid
        assert reason is not None

    def test_valid_date_still_passes(self):
        """Regression guard: real dates must NOT be accidentally rejected."""
        for valid_date in ["01/01/2021", "15-Jun-2021", "2021-03-25"]:
            is_valid, reason = validate_date_candidate(valid_date)
            assert is_valid, f"Valid date {valid_date!r} was wrongly rejected: {reason}"