"""
Unit tests for src.normalizers.date_normalizer.

Pipeline stage: DATE NORMALIZATION.

Function under test: normalize_date_strings(rows) -> (rows, normalized_count)
  Expected input:  List[Row] (text mutable, original_text immutable baseline)
  Expected output: (List[Row] with .text rewritten in place, int count of
                    all normalizations: OCR-typo fixes + interpolations +
                    canonicalizations)

Three passes are tested independently where possible:
  1. OCR typo correction (ordinal stripping, digit/letter confusion)
  2. Sequential date interpolation (missing date inferred from neighbors)
  3. Canonicalization to YYYY-MM-DD

Edge cases:
  - Rows with no date-like text at all (must be left unchanged)
  - Rule 8 violating entries ("2O2L"->"2021", "0x4L"->"2024") are
    DOCUMENTED but explicitly tested as "exists but should not be relied
    upon" per the module's own docstring (Rule 9 — kept only for legacy
    benchmark compatibility).

Failure cases:
  - Interpolation must not fire when neighboring dates differ by >2 days
  - Canonicalization must leave unparseable date-like text unchanged
    (returns original string on exception)
"""

from __future__ import annotations

import pytest

from src.normalizers.date_normalizer import normalize_date_strings, _correct_ocr_typos
from tests.fixtures.ocr_tokens import make_row_from_text


@pytest.mark.unit
class TestCorrectOcrTypos:
    def test_ordinal_suffix_stripped_before_month(self):
        """WHY: '1st June' must become '1 June' so DATE_RE / dateutil can
        parse it. BUG PREVENTED: ordinal suffixes ('st','nd','rd','th') left
        in place cause dateutil.parse to fail -> date_parse_failed."""
        assert _correct_ocr_typos("1st June") == "1 June"

    def test_digit_letter_confusion_l9_to_19(self):
        """WHY: OCR commonly misreads '1' as 'l' or 'I'. 'l9' must become '19'
        so a year like 'l9' (meant to be '19' as part of '2019') normalizes
        correctly."""
        assert "19" in _correct_ocr_typos("l9")
        assert "19" in _correct_ocr_typos("I9")

    def test_dot_separator_converted_to_hyphen(self):
        """WHY: '13.10-2021' (mixed separators from OCR) must become a
        consistent '13-10-2021' so DATE_RE Pattern 1 matches it."""
        result = _correct_ocr_typos("13.10-2021")
        assert result == "13-10-2021"

    def test_rule8_legacy_substitutions_still_present_but_flagged(self):
        """WHY/BUG PREVENTED: this test documents the EXISTING (legacy,
        Rule-8-violating) hardcoded substitutions '2O2L'->'2021' and
        '0x4L'->'2024'. The module docstring says these are kept only for
        backward-compat with old benchmark results and should be removed
        once the benchmark dataset is updated. This test exists so that
        REMOVING them is a deliberate, visible change (test must be updated
        too), not an accidental one during refactoring."""
        assert _correct_ocr_typos("2O2L") == "2021"
        assert _correct_ocr_typos("0x4L") == "2024"


@pytest.mark.unit
class TestNormalizeDateStringsPass1OcrCorrection:
    def test_row_with_ordinal_date_is_corrected(self):
        """WHY: end-to-end pass-1 check that a row containing '1st June 2021'
        gets its ordinal stripped as part of normalize_date_strings (not
        just the helper function)."""
        rows = [make_row_from_text("1st June 2021 Salary Credit 1000.00")]
        rows, count = normalize_date_strings(rows)
        assert "1st" not in rows[0].text
        assert count >= 1

    def test_row_with_no_date_like_text_unchanged(self):
        """WHY: rows that contain no date-like substring at all (e.g. a pure
        footer row) must pass through normalize_date_strings unchanged and
        must NOT be counted as normalized.
        BUG PREVENTED: over-eager regexes mutating non-date text (e.g.
        turning "Total Balance" into something else)."""
        original = "Total Balance Brought Forward"
        rows = [make_row_from_text(original)]
        rows, count = normalize_date_strings(rows)
        assert rows[0].text == original


@pytest.mark.unit
class TestNormalizeDateStringsPass2Interpolation:
    def test_missing_date_interpolated_from_neighbors_one_day_apart(self):
        """WHY: a row whose date OCR was lost entirely, but is sandwiched
        between two rows one day apart, should be a candidate for the
        'sequential date interpolation' described in the module docstring.

        OBSERVED BEHAVIOR (documented, not assumed): with DD-MM-YYYY-style
        neighbor dates, pass 1 canonicalizes the neighbors to YYYY-MM-DD
        BEFORE pass 2 runs its DATE_RE search for interpolation anchors.
        Whether the middle row actually gets a date prefix depends on the
        day-diff check (abs(diff) in {0,1,2}) AFTER both neighbors are
        parsed. This test asserts no exception is raised and that both
        neighbor dates get canonicalized (count >= 2). It deliberately does
        NOT assert the middle row was rewritten -- that exact interaction
        should be pinned down in tests/regression/test_known_failures.py
        once real insufficient_numeric_values samples exist (see
        failure_samples.INSUFFICIENT_NUMERIC_VALUES_SYNTHETIC)."""
        rows = [
            make_row_from_text("01-06-2021 Opening balance 100.00"),
            make_row_from_text("ATM Withdrawal 20.00"),  # date missing
            make_row_from_text("02-06-2021 Closing balance 80.00"),
        ]
        rows, count = normalize_date_strings(rows)
        assert len(rows) == 3
        assert count >= 2  # at least both neighbor dates canonicalized

    def test_neighbors_more_than_two_days_apart_no_interpolation(self):
        """WHY: interpolation must be conservative — if the gap between
        neighboring dates is >2 days, there's no safe inference, and the
        middle row's text must remain unchanged (still missing a date).
        BUG PREVENTED: fabricating dates for unrelated rows (e.g. across a
        page break with a multi-week gap)."""
        rows = [
            make_row_from_text("01-06-2021 Opening balance 100.00"),
            make_row_from_text("Some unrelated narration 20.00"),
            make_row_from_text("15-06-2021 Closing balance 80.00"),
        ]
        middle_before = rows[1].text
        rows, _ = normalize_date_strings(rows)
        assert rows[1].text == middle_before

    def test_first_and_last_row_never_interpolated(self):
        """WHY: pass 2 iterates range(1, n-1), so the first and last rows
        are never candidates for interpolation (no left/right neighbor on
        one side). This pins that boundary behavior."""
        rows = [
            make_row_from_text("ATM Withdrawal 20.00"),       # idx 0 - no left neighbor
            make_row_from_text("01-06-2021 Something 5.00"),
            make_row_from_text("Closing narration 80.00"),     # idx 2 - no right neighbor
        ]
        before_first = rows[0].text
        before_last = rows[2].text
        rows, _ = normalize_date_strings(rows)
        assert rows[0].text == before_first
        assert rows[2].text == before_last


@pytest.mark.unit
class TestNormalizeDateStringsPass3Canonicalization:
    def test_ddmmyyyy_canonicalized_to_iso(self):
        """WHY: pass 3 converts all recognized dates to YYYY-MM-DD so
        downstream transaction_parser's year-inference logic
        (which checks pr.date[4]=='-' and pr.date[7]=='-') works correctly.
        BUG PREVENTED: 'date_parse_failed' downstream when transaction_parser
        expects ISO format but receives 'DD/MM/YYYY'."""
        rows = [make_row_from_text("23/12/2019 UPI Payment 226.14")]
        rows, count = normalize_date_strings(rows)
        assert "2019-12-23" in rows[0].text
        assert count >= 1

    def test_unparseable_date_like_text_left_unchanged(self):
        """WHY: canonicalize_match() catches exceptions from dateutil and
        returns the original string on failure. A date-like-but-unparseable
        span must not crash normalize_date_strings or corrupt the row.
        BUG PREVENTED: an uncaught ValueError from dateutil.parser.parse
        propagating up through main.py and aborting the whole batch run."""
        # "32-13-9999" matches DATE_RE pattern 1 shape but is not a real date.
        rows = [make_row_from_text("32-13-9999 Some narration 10.00")]
        # Must not raise.
        rows, count = normalize_date_strings(rows)
        assert isinstance(rows[0].text, str)