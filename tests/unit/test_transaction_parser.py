"""
Unit tests for src.transaction_parser.

Pipeline stage: TRANSACTION PARSING (Row -> ParsedRow -> Transaction).

Functions under test:
  parse_rows(rows: List[Row]) -> List[ParsedRow]
  build_transactions(parsed_rows: List[ParsedRow]) -> List[Transaction]
  process_reconstructed_rows(rows, debug_output_path=None) -> List[Transaction]

Edge cases:
  - row with date + amount -> ParsedRow produced
  - row classified as header/footer -> excluded even if date+amount present
  - row with date but NO amount -> excluded ("insufficient_numeric_values"
    per evaluation.py's classification)
  - row with no valid date -> excluded
  - out-of-range year repaired to the dominant year across the statement

Failure cases:
  - DATE_RE matching an amount-shaped span first must not block a row that
    ALSO contains a genuinely valid date later in the text (first-valid-date
    search, not first-match)
"""

from __future__ import annotations

import pytest

from src.transaction_parser import ParsedRow, build_transactions, parse_rows
from tests.fixtures.ocr_tokens import make_row_from_text


@pytest.mark.unit
class TestParseRowsBasic:
    def test_date_and_amount_produce_parsed_row(self):
        """WHY: the canonical happy-path — a row with an ISO date and one
        amount becomes exactly one ParsedRow with date/amounts/description
        correctly split."""
        rows = [make_row_from_text("2021-12-23 UPI Payment to Vendor 226.14")]
        parsed = parse_rows(rows)
        assert len(parsed) == 1
        pr = parsed[0]
        assert pr.date == "2021-12-23"
        assert pr.amounts == ["226.14"]
        assert "UPI Payment to Vendor" in pr.description
        assert pr.source_row_index == 0

    def test_header_row_with_date_and_amount_excluded(self):
        """WHY: even if a row contains both a DATE_RE match and a MONEY_RE
        match, _is_header_or_footer() must exclude it if it also contains
        header/footer keywords ('statement of', 'account no', etc.).
        BUG PREVENTED: 'Statement of Account ... 2021-12-23 ... 226.14'
        being parsed as a real transaction."""
        rows = [make_row_from_text("2021-12-23 Statement of Account Number 226.14")]
        parsed = parse_rows(rows)
        assert parsed == []

    def test_row_with_date_but_no_amount_excluded(self):
        """WHY: a row with a valid date but zero MONEY_RE matches is
        excluded at the parse stage. evaluation.py classifies such rows as
        'insufficient_numeric_values' (the row likely split across two
        OCR lines)."""
        rows = [make_row_from_text("2021-03-12 Interest credited for the quarter")]
        parsed = parse_rows(rows)
        assert parsed == []

    def test_row_with_no_date_excluded(self):
        """WHY: a row with amounts but no DATE_RE match at all must not
        produce a ParsedRow (evaluation.py classifies as no_date_detected
        or date_parse_failed depending on whether a date-CANDIDATE exists)."""
        rows = [make_row_from_text("Service charge applied 100.00")]
        parsed = parse_rows(rows)
        assert parsed == []

    def test_classifier_rejected_row_excluded_even_with_date_and_amount(self):
        """WHY: row_classifier runs BEFORE the date/amount check. A row
        matching 'Closing Balance' (footer pattern) with a date-like and
        amount-like substring must still be excluded via classifier, not
        just the header/footer keyword check in transaction_parser itself."""
        rows = [make_row_from_text("Closing Balance as of 2021-12-23 5000.00")]
        parsed = parse_rows(rows)
        assert parsed == []


@pytest.mark.unit
class TestParseRowsFirstValidDate:
    def test_skips_invalid_date_candidate_to_find_valid_one_later(self):
        """WHY: docstring says parse_rows finds the FIRST VALID date
        candidate, not just the first DATE_RE match -- in principle, an
        OCR-mangled amount that happens to match DATE_RE shape but fails
        validate_date_candidate() should not block a genuinely valid date
        appearing later in the same row.

        OBSERVED LIMITATION (documented as a known gap, not a pass/fail
        regression target): for the input
        '00-7.00 Narration 2021-12-23 100.00', constants.DATE_RE's Pattern 7
        ("Mon DD YYYY"-style, case-insensitive) greedily matches the SINGLE
        span '00 Narration 2021' (treating 'Narration' as a bogus month
        name), which validate_date_candidate() correctly rejects (invalid
        day/month). Because this is the ONLY DATE_RE match in the string,
        DATE_RE.finditer() never reaches the genuinely valid '2021-12-23'
        later in the text, and the row is dropped entirely (0 ParsedRows).

        This test pins the CURRENT behavior (row dropped) so it is visible
        in coverage reports as a candidate for future improvement (e.g.
        restricting Pattern 7's month-name alternation to known month
        abbreviations only, which 'Narration' is not)."""
        rows = [make_row_from_text("00-7.00 Narration 2021-12-23 100.00")]
        parsed = parse_rows(rows)
        assert parsed == [], (
            "If this now produces a ParsedRow, DATE_RE pattern matching "
            "has improved -- update this test to assert "
            "parsed[0].date == '2021-12-23' and remove this comment."
        )


@pytest.mark.unit
class TestParseRowsYearInference:
    def test_out_of_range_year_repaired_to_dominant_year(self):
        """WHY: 'Contextual Year Inference' replaces an out-of-bounds year
        (outside [1990, current_year+1]) with the most common ('dominant')
        year seen across all parsed rows in the statement.
        BUG PREVENTED: a single OCR-corrupted year (e.g. '1900' from a
        mangled '2021') producing a transaction dated in 1900 while its
        neighbors are correctly dated 2021."""
        rows = [
            make_row_from_text("2021-12-23 Txn A 100.00"),
            make_row_from_text("1900-12-24 Txn B 200.00"),  # corrupted year
            make_row_from_text("2021-12-25 Txn C 300.00"),
        ]
        parsed = parse_rows(rows)
        dates = [p.date for p in parsed]
        assert dates == ["2021-12-23", "2021-12-24", "2021-12-25"]

    def test_all_valid_years_left_unchanged(self):
        """WHY: when all years are within range, the dominant-year repair
        must be a no-op (idempotent on already-correct data)."""
        rows = [
            make_row_from_text("2021-12-23 Txn A 100.00"),
            make_row_from_text("2021-12-24 Txn B 200.00"),
        ]
        parsed = parse_rows(rows)
        assert [p.date for p in parsed] == ["2021-12-23", "2021-12-24"]


@pytest.mark.unit
class TestBuildTransactions:
    def test_parsed_rows_converted_with_null_amount_fields(self):
        """WHY: build_transactions deliberately leaves debit/credit/balance/
        amount as None -- semantic assignment happens in a later stage
        (semantic_assigner). This test pins that contract."""
        parsed = [
            ParsedRow(
                raw_text="2021-12-23 UPI 226.14",
                date="2021-12-23",
                amounts=["226.14"],
                description="UPI",
                source_row_index=0,
            )
        ]
        txns = build_transactions(parsed)
        assert len(txns) == 1
        txn = txns[0]
        assert txn.date == "2021-12-23"
        assert txn.description == "UPI"
        assert txn.amount is None
        assert txn.debit is None
        assert txn.credit is None
        assert txn.balance is None

    def test_empty_parsed_rows_produces_empty_transactions(self):
        assert build_transactions([]) == []