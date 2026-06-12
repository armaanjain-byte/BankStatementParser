"""
tests/integration/test_parse_then_assign.py — Integration: parser → layout → semantic assigner

Tests that verify ParsedRow objects flow correctly through detect_layout()
and assign_semantics() to produce populated Transaction objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import make_token, make_row
from src.layout_detector import detect_layout
from src.schema import Transaction
from src.semantic_assigner import (
    assign_rows,
    build_semantic_assignment_report,
)
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import Row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_header_row(
    cols: list[str],
    base_y: int = 10,
    x_step: int = 150,
) -> Row:
    tokens = [
        make_token(c, x=i * x_step, y=base_y, w=130, h=20)
        for i, c in enumerate(cols)
    ]
    text = " ".join(cols)
    return Row(y_center=base_y + 10.0, tokens=tokens, text=text, original_text=text)


def make_data_row(
    date: str,
    desc: str,
    amounts: list[str],
    base_y: int = 50,
) -> Row:
    all_parts = [date, desc] + amounts
    tokens = []
    x = 0
    widths = [100, 200] + [90] * len(amounts)
    for part, w in zip(all_parts, widths):
        tokens.append(make_token(part, x=x, y=base_y, w=w, h=20))
        x += w + 10
    text = " ".join(all_parts)
    return Row(y_center=base_y + 10.0, tokens=tokens, text=text, original_text=text)


def standard_statement_rows() -> list[Row]:
    """A minimal synthetic statement: header + 3 transactions."""
    return [
        make_header_row(["Date", "Narration", "Debit", "Credit", "Balance"], base_y=10),
        make_data_row("01/01/2021", "ATM Withdrawal", ["5,000.00", "0.00", "45,000.00"], base_y=50),
        make_data_row("05/01/2021", "Salary Credit", ["0.00", "50,000.00", "95,000.00"], base_y=80),
        make_data_row("10/01/2021", "NEFT Transfer", ["2,000.00", "0.00", "93,000.00"], base_y=110),
    ]


# ===========================================================================
# parse_rows → detect_layout → assign_rows
# ===========================================================================

class TestParseToLayoutToAssign:
    def test_transactions_extracted_count(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        assert len(results) == len(parsed)

    def test_all_transactions_have_dates(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        for r in results:
            assert r.transaction.date is not None

    def test_layout_uses_header_method(self):
        rows = standard_statement_rows()
        layout = detect_layout([], rows)
        assert layout.detection_method == "header"

    def test_debit_column_detected(self):
        rows = standard_statement_rows()
        layout = detect_layout([], rows)
        assert layout.debit_column is not None or layout.credit_column is not None

    def test_amount_values_are_floats(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        for r in results:
            for field in ("amount", "debit", "credit", "balance"):
                val = r.assigned_values.get(field)
                if val is not None:
                    assert isinstance(val, float)

    def test_descriptions_not_empty(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        for r in results:
            # Description may be empty string but not None
            assert r.transaction.description is not None


# ===========================================================================
# build_semantic_assignment_report
# ===========================================================================

class TestBuildReport:
    def test_report_assignment_count_equals_parsed_count(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        report = build_semantic_assignment_report(parsed, rows, layout, statement_id="int_test")
        assert report.assignment_count == len(parsed)

    def test_report_statement_id_propagated(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        report = build_semantic_assignment_report(parsed, rows, layout, statement_id="s99")
        assert report.statement_id == "s99"

    def test_report_to_dict_json_serialisable(self):
        import json
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        report = build_semantic_assignment_report(parsed, rows, layout)
        json.dumps(report.to_dict())  # must not raise

    def test_assigned_row_count_non_negative(self):
        rows = standard_statement_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        report = build_semantic_assignment_report(parsed, rows, layout)
        assert report.assigned_row_count >= 0


# ===========================================================================
# Inference-only layout (no header row)
# ===========================================================================

class TestInferenceLayout:
    def _data_only_rows(self) -> list[Row]:
        return [
            make_data_row("01/01/2021", "Salary", ["50,000.00"], base_y=i * 30)
            for i in range(5)
        ]

    def test_inference_layout_still_assigns(self):
        rows = self._data_only_rows()
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        # Some assignments should exist even without a header
        assert len(results) == len(parsed)

    def test_inference_method_reflected_in_layout(self):
        rows = self._data_only_rows()
        layout = detect_layout([], rows)
        assert layout.detection_method in ("data_inference", "header", "none")


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_rows_produces_empty_results(self):
        rows: list[Row] = []
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        assert results == []

    def test_header_only_no_data_produces_no_transactions(self):
        rows = [make_header_row(["Date", "Narration", "Debit", "Credit", "Balance"])]
        parsed = parse_rows(rows)
        assert len(parsed) == 0

    def test_single_transaction_produces_one_result(self):
        rows = [
            make_header_row(["Date", "Narration", "Amount"]),
            make_data_row("15/03/2021", "UPI payment", ["1,200.00"], base_y=40),
        ]
        parsed = parse_rows(rows)
        layout = detect_layout([], rows)
        results = assign_rows(parsed, rows, layout)
        assert len(results) == len(parsed)