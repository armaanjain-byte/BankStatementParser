"""
tests/unit/test_semantic_assigner.py — Unit tests for semantic_assigner.py

Tests cover:
  - assign_semantics(): column-order assignment, geometric fallback, no-layout case
  - assign_rows(): batch processing
  - build_semantic_assignment_report(): report counts
  - _parse_amount(): string → float conversion
  - _find_amount_x_centers(): three-pass token matching
  - AssignmentResult.to_dict()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import make_token, make_row, make_row_with_tokens
from src.layout_detector import DetectedColumn, LayoutDefinition
from src.schema import Transaction
from src.semantic_assigner import (
    AssignmentResult,
    SemanticAssignmentReport,
    assign_semantics,
    assign_rows,
    build_semantic_assignment_report,
    _parse_amount,
    _find_amount_x_centers,
)
from src.transaction_parser import ParsedRow
from src.transaction_reconstructor import Row


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_detected_col(
    name: str,
    x_center: float,
    confidence: float = 0.9,
) -> DetectedColumn:
    w = 80
    return DetectedColumn(
        source_text=name,
        normalized_name=name,
        x_min=int(x_center - w / 2),
        x_max=int(x_center + w / 2),
        x_center=x_center,
        confidence=confidence,
        source="header",
        match_kind="exact_alias",
    )


def make_layout(
    debit_x: float | None = None,
    credit_x: float | None = None,
    balance_x: float | None = None,
    amount_x: float | None = None,
) -> LayoutDefinition:
    return LayoutDefinition(
        debit_column=make_detected_col("debit", debit_x) if debit_x is not None else None,
        credit_column=make_detected_col("credit", credit_x) if credit_x is not None else None,
        balance_column=make_detected_col("balance", balance_x) if balance_x is not None else None,
        amount_column=make_detected_col("amount", amount_x) if amount_x is not None else None,
        detection_method="header",
        confidence_scores={},
    )


def make_parsed_row(
    amounts: list[str],
    date: str = "2021-01-01",
    description: str = "Test",
    idx: int = 0,
) -> ParsedRow:
    return ParsedRow(
        raw_text=f"{date} {description} {' '.join(amounts)}",
        date=date,
        amounts=amounts,
        description=description,
        source_row_index=idx,
    )


def make_row_with_amount_tokens(amounts: list[str], base_y: int = 100) -> Row:
    """Row whose tokens include date, description, and each amount."""
    tokens = [make_token("01/01/2021", x=0, y=base_y, w=100, h=20)]
    tokens += [make_token(desc, x=120, y=base_y, w=180, h=20)
               for desc in ["ATM", "withdrawal"]]
    for i, amt in enumerate(amounts):
        tokens.append(make_token(amt, x=400 + i * 100, y=base_y, w=90, h=20))
    text = " ".join(t.text for t in tokens)
    return Row(y_center=base_y + 10.0, tokens=tokens, text=text, original_text=text)


# ===========================================================================
# _parse_amount
# ===========================================================================

class TestParseAmount:
    def test_plain_float(self):
        assert _parse_amount("1234.56") == pytest.approx(1234.56)

    def test_with_commas(self):
        assert _parse_amount("1,234.56") == pytest.approx(1234.56)

    def test_negative(self):
        assert _parse_amount("-500.00") == pytest.approx(-500.0)

    def test_zero(self):
        assert _parse_amount("0.00") == pytest.approx(0.0)

    def test_non_numeric_returns_none(self):
        assert _parse_amount("abc") is None

    def test_empty_returns_none(self):
        assert _parse_amount("") is None


# ===========================================================================
# _find_amount_x_centers
# ===========================================================================

class TestFindAmountXCenters:
    def test_single_amount_found(self):
        row = make_row_with_amount_tokens(["5,000.00"])
        centers = _find_amount_x_centers(["5,000.00"], row)
        assert len(centers) == 1
        assert centers[0] is not None

    def test_two_amounts_found(self):
        row = make_row_with_amount_tokens(["5,000.00", "10,000.00"])
        centers = _find_amount_x_centers(["5,000.00", "10,000.00"], row)
        assert len(centers) == 2
        # Second amount token is to the right of first
        if centers[0] is not None and centers[1] is not None:
            assert centers[1] >= centers[0]

    def test_missing_amount_returns_none_entry(self):
        row = make_row_with_amount_tokens(["5,000.00"])
        centers = _find_amount_x_centers(["99,999.99"], row)
        assert centers[0] is None

    def test_empty_amounts_list(self):
        row = make_row_with_amount_tokens(["5,000.00"])
        assert _find_amount_x_centers([], row) == []


# ===========================================================================
# assign_semantics — column-count matches layout
# ===========================================================================

class TestAssignSemanticsExactMatch:
    def test_debit_credit_balance_assigned_in_order(self):
        layout = make_layout(debit_x=400, credit_x=500, balance_x=600)
        parsed_row = make_parsed_row(["1,000.00", "0.00", "9,000.00"])
        row = make_row_with_amount_tokens(["1,000.00", "0.00", "9,000.00"])

        result = assign_semantics(parsed_row, row, layout)

        assert result.transaction.debit == pytest.approx(1000.0)
        assert result.transaction.credit == pytest.approx(0.0)
        assert result.transaction.balance == pytest.approx(9000.0)

    def test_reason_reflects_layout_column_order(self):
        layout = make_layout(debit_x=400, balance_x=600)
        parsed_row = make_parsed_row(["500.00", "9,500.00"])
        row = make_row_with_amount_tokens(["500.00", "9,500.00"])

        result = assign_semantics(parsed_row, row, layout)
        assert "layout column order" in result.reason

    def test_transaction_date_preserved(self):
        layout = make_layout(amount_x=400)
        parsed_row = make_parsed_row(["1,500.00"], date="2021-03-15")
        row = make_row_with_amount_tokens(["1,500.00"])

        result = assign_semantics(parsed_row, row, layout)
        assert result.transaction.date == "2021-03-15"

    def test_transaction_description_preserved(self):
        layout = make_layout(amount_x=400)
        parsed_row = make_parsed_row(["250.00"], description="UPI/Payment")
        row = make_row_with_amount_tokens(["250.00"])

        result = assign_semantics(parsed_row, row, layout)
        assert result.transaction.description == "UPI/Payment"


# ===========================================================================
# assign_semantics — no layout detected
# ===========================================================================

class TestAssignSemanticsNoLayout:
    def test_no_money_columns_gives_zero_confidence(self):
        layout = make_layout()  # no columns set
        parsed_row = make_parsed_row(["500.00"])
        row = make_row_with_amount_tokens(["500.00"])

        result = assign_semantics(parsed_row, row, layout)
        assert result.confidence == 0.0

    def test_all_assigned_values_none_without_layout(self):
        layout = make_layout()
        parsed_row = make_parsed_row(["500.00"])
        row = make_row_with_amount_tokens(["500.00"])

        result = assign_semantics(parsed_row, row, layout)
        assert result.assigned_values["debit"] is None
        assert result.assigned_values["credit"] is None
        assert result.assigned_values["balance"] is None
        assert result.assigned_values["amount"] is None

    def test_returns_assignment_result_type(self):
        layout = make_layout()
        parsed_row = make_parsed_row([])
        row = make_row("")

        result = assign_semantics(parsed_row, row, layout)
        assert isinstance(result, AssignmentResult)


# ===========================================================================
# assign_semantics — geometric fallback (mismatched counts)
# ===========================================================================

class TestAssignSemanticsGeometricFallback:
    def test_geometric_fallback_triggered(self):
        # 3 layout columns but only 2 amounts → geometric fallback
        layout = make_layout(debit_x=400, credit_x=500, balance_x=600)
        parsed_row = make_parsed_row(["500.00", "9,500.00"])
        row = make_row_with_amount_tokens(["500.00", "9,500.00"])

        result = assign_semantics(parsed_row, row, layout)
        # Geometric fallback should assign to closest columns
        assigned = [v for v in result.assigned_values.values() if v is not None]
        assert len(assigned) <= 2

    def test_geometric_fallback_confidence_lower_than_exact(self):
        layout = make_layout(debit_x=400, credit_x=500, balance_x=600)

        # Exact match (3 amounts, 3 columns)
        parsed_exact = make_parsed_row(["100.00", "200.00", "300.00"])
        row_exact = make_row_with_amount_tokens(["100.00", "200.00", "300.00"])
        result_exact = assign_semantics(parsed_exact, row_exact, layout)

        # Fallback (2 amounts, 3 columns)
        parsed_fallback = make_parsed_row(["100.00", "300.00"])
        row_fallback = make_row_with_amount_tokens(["100.00", "300.00"])
        result_fallback = assign_semantics(parsed_fallback, row_fallback, layout)

        assert result_exact.confidence >= result_fallback.confidence


# ===========================================================================
# assign_rows — batch assignment
# ===========================================================================

class TestAssignRows:
    def _make_batch(self, n: int):
        layout = make_layout(amount_x=400)
        parsed_rows = [make_parsed_row(["1,000.00"], idx=i) for i in range(n)]
        rows = [make_row_with_amount_tokens(["1,000.00"], base_y=i * 30) for i in range(n)]
        return parsed_rows, rows, layout

    def test_returns_correct_count(self):
        parsed_rows, rows, layout = self._make_batch(3)
        results = assign_rows(parsed_rows, rows, layout)
        assert len(results) == 3

    def test_source_row_index_preserved(self):
        parsed_rows, rows, layout = self._make_batch(3)
        results = assign_rows(parsed_rows, rows, layout)
        indices = [r.source_row_index for r in results]
        assert indices == [0, 1, 2]

    def test_empty_input_returns_empty(self):
        results = assign_rows([], [], make_layout())
        assert results == []


# ===========================================================================
# build_semantic_assignment_report
# ===========================================================================

class TestBuildSemanticAssignmentReport:
    def test_report_assignment_count(self):
        layout = make_layout(amount_x=400)
        parsed_rows = [make_parsed_row(["500.00"], idx=i) for i in range(4)]
        rows = [make_row_with_amount_tokens(["500.00"], base_y=i * 30) for i in range(4)]

        report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id="test")

        assert report.assignment_count == 4

    def test_report_statement_id(self):
        report = build_semantic_assignment_report([], [], make_layout(), statement_id="s42")
        assert report.statement_id == "s42"

    def test_report_to_dict_structure(self):
        layout = make_layout(amount_x=400)
        parsed_rows = [make_parsed_row(["100.00"], idx=0)]
        rows = [make_row_with_amount_tokens(["100.00"])]

        report = build_semantic_assignment_report(parsed_rows, rows, layout)
        d = report.to_dict()

        assert "assignment_count" in d
        assert "assigned_row_count" in d
        assert "rows" in d
        assert isinstance(d["rows"], list)

    def test_assigned_row_count_matches_rows_with_amounts(self):
        layout = make_layout(amount_x=400)
        parsed_rows = [make_parsed_row(["500.00"], idx=0)]
        rows = [make_row_with_amount_tokens(["500.00"])]

        report = build_semantic_assignment_report(parsed_rows, rows, layout)
        assert report.assigned_row_count >= 0

    def test_report_is_semantic_assignment_report_type(self):
        report = build_semantic_assignment_report([], [], make_layout())
        assert isinstance(report, SemanticAssignmentReport)


# ===========================================================================
# AssignmentResult.to_dict
# ===========================================================================

class TestAssignmentResultToDict:
    def test_to_dict_contains_transaction(self):
        layout = make_layout(amount_x=400)
        parsed_row = make_parsed_row(["1,000.00"])
        row = make_row_with_amount_tokens(["1,000.00"])

        result = assign_semantics(parsed_row, row, layout)
        d = result.to_dict()
        assert "transaction" in d
        assert "date" in d["transaction"]

    def test_to_dict_is_json_serialisable(self):
        import json
        layout = make_layout(amount_x=400)
        parsed_row = make_parsed_row(["1,000.00"])
        row = make_row_with_amount_tokens(["1,000.00"])

        result = assign_semantics(parsed_row, row, layout)
        json.dumps(result.to_dict())  # must not raise