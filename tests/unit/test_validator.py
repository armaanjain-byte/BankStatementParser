"""
tests/unit/test_validator.py — Unit tests for validator.py

Tests cover:
  - validate_statement(): no-transaction warning
  - validate_statement(): no_layout_detected / layout_inferred issues
  - validate_statement(): missing_date error per transaction
  - validate_statement(): unassigned_amounts warning
  - validate_statement(): missing_account_number warning
  - validate_statement(): pass/fail based on ERROR severity
  - _calculate_confidence(): score range and penalty logic
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.layout_detector import DetectedColumn, LayoutDefinition
from src.models.transaction_models import BankStatement, StatementMetadata, Transaction
from src.models.validation_models import ValidationReport, ValidationIssue
from src.semantic_assigner import AssignmentResult, SemanticAssignmentReport
from src.validator import validate_statement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detected_col(name: str, x: float = 400.0) -> DetectedColumn:
    return DetectedColumn(
        source_text=name,
        normalized_name=name,
        x_min=int(x - 40),
        x_max=int(x + 40),
        x_center=x,
        confidence=0.9,
        source="header",
        match_kind="exact_alias",
    )


def make_layout(
    detection_method: str = "header",
    has_balance_col: bool = False,
    has_amount_col: bool = True,
) -> LayoutDefinition:
    return LayoutDefinition(
        detection_method=detection_method,
        balance_column=make_detected_col("balance", 600) if has_balance_col else None,
        amount_column=make_detected_col("amount", 400) if has_amount_col else None,
        confidence_scores={"table_header": 0.85},
    )


def make_assignment_result(
    txn: Transaction,
    raw_amounts: list[str] | None = None,
    source_row_index: int = 0,
) -> AssignmentResult:
    has_amt = txn.amount is not None or txn.debit is not None or txn.credit is not None
    return AssignmentResult(
        source_row_index=source_row_index,
        raw_text="test row",
        raw_amounts=raw_amounts or [],
        layout_used={},
        assigned_values={
            "amount": txn.amount,
            "debit": txn.debit,
            "credit": txn.credit,
            "balance": txn.balance,
        },
        confidence=0.85 if has_amt else 0.0,
        reason="assigned by layout" if has_amt else "no layout",
        transaction=txn,
    )


def make_statement(transactions: list[Transaction], account_number: str | None = "12345") -> BankStatement:
    return BankStatement(
        metadata=StatementMetadata(account_number=account_number),
        transactions=transactions,
    )


def make_report(results: list[AssignmentResult], statement_id: str | None = None) -> SemanticAssignmentReport:
    assigned = sum(
        1 for r in results
        if any(v is not None for v in r.assigned_values.values())
    )
    return SemanticAssignmentReport(
        statement_id=statement_id,
        layout_summary={},
        assignment_count=len(results),
        assigned_row_count=assigned,
        rows=results,
    )


def txn(date: str = "2021-01-01", amount: float | None = 1000.0) -> Transaction:
    return Transaction(date=date, description="Test", amount=amount)


# ===========================================================================
# validate_statement — basic output type
# ===========================================================================

class TestValidateStatementReturnType:
    def test_returns_validation_report(self):
        statement = make_statement([txn()])
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        assert isinstance(result, ValidationReport)

    def test_passed_is_bool(self):
        statement = make_statement([txn()])
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        assert isinstance(result.passed, bool)

    def test_confidence_in_valid_range(self):
        statement = make_statement([txn()])
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# No transactions
# ===========================================================================

class TestNoTransactions:
    def test_no_transactions_adds_warning(self):
        statement = make_statement([])
        layout = make_layout()
        report = make_report([])
        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "no_transactions" in types

    def test_no_transactions_warning_severity(self):
        statement = make_statement([])
        layout = make_layout()
        report = make_report([])
        result = validate_statement(statement, layout, report)
        for issue in result.issues:
            if issue.issue_type == "no_transactions":
                assert issue.severity == "WARNING"

    def test_no_transactions_confidence_zero(self):
        statement = make_statement([])
        layout = make_layout()
        report = make_report([])
        result = validate_statement(statement, layout, report)
        assert result.confidence == 0.0


# ===========================================================================
# Layout detection method
# ===========================================================================

class TestLayoutDetectionMethod:
    def test_no_layout_adds_error(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="none")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "no_layout_detected" in types

    def test_no_layout_error_severity(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="none")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        for issue in result.issues:
            if issue.issue_type == "no_layout_detected":
                assert issue.severity == "ERROR"

    def test_no_layout_fails_validation(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="none")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        assert result.passed is False

    def test_data_inference_adds_warning(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="data_inference")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "layout_inferred" in types

    def test_data_inference_warning_severity(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="data_inference")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        for issue in result.issues:
            if issue.issue_type == "layout_inferred":
                assert issue.severity == "WARNING"

    def test_header_detection_no_layout_issue(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="header")
        report = make_report([make_assignment_result(txn())])
        result = validate_statement(statement, layout, report)
        layout_issues = [
            i for i in result.issues
            if i.issue_type in ("no_layout_detected", "layout_inferred")
        ]
        assert len(layout_issues) == 0


# ===========================================================================
# Missing date
# ===========================================================================

class TestMissingDate:
    def test_missing_date_adds_error(self):
        bad_txn = Transaction(date=None, description="Unknown", amount=500.0)
        statement = make_statement([bad_txn])
        layout = make_layout()
        result_obj = make_assignment_result(bad_txn, raw_amounts=["500.00"])
        report = make_report([result_obj])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "missing_date" in types

    def test_missing_date_error_severity(self):
        bad_txn = Transaction(date=None, description="Unknown", amount=500.0)
        statement = make_statement([bad_txn])
        layout = make_layout()
        report = make_report([make_assignment_result(bad_txn)])

        result = validate_statement(statement, layout, report)
        for issue in result.issues:
            if issue.issue_type == "missing_date":
                assert issue.severity == "ERROR"

    def test_missing_date_fails_validation(self):
        bad_txn = Transaction(date=None, description="Test", amount=100.0)
        statement = make_statement([bad_txn])
        layout = make_layout()
        report = make_report([make_assignment_result(bad_txn)])

        result = validate_statement(statement, layout, report)
        assert result.passed is False

    def test_valid_date_no_error(self):
        good_txn = txn(date="2021-01-01")
        statement = make_statement([good_txn])
        layout = make_layout()
        report = make_report([make_assignment_result(good_txn)])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "missing_date" not in types


# ===========================================================================
# Unassigned amounts
# ===========================================================================

class TestUnassignedAmounts:
    def test_unassigned_amounts_adds_warning(self):
        # Transaction with raw amounts but nothing assigned to semantic fields
        unassigned_txn = Transaction(date="2021-01-01", description="Test",
                                     amount=None, debit=None, credit=None, balance=None)
        result_obj = make_assignment_result(unassigned_txn, raw_amounts=["500.00"])
        # Layout has a money column so assignment was expected
        layout = make_layout(has_amount_col=True)
        statement = make_statement([unassigned_txn])
        report = make_report([result_obj])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "unassigned_amounts" in types

    def test_no_unassigned_warning_when_all_assigned(self):
        good_txn = txn(amount=1000.0)
        statement = make_statement([good_txn])
        layout = make_layout()
        report = make_report([make_assignment_result(good_txn, raw_amounts=["1,000.00"])])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "unassigned_amounts" not in types


# ===========================================================================
# Missing account number
# ===========================================================================

class TestMissingAccountNumber:
    def test_missing_account_number_adds_warning(self):
        statement = make_statement([txn()], account_number=None)
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "missing_account_number" in types

    def test_missing_account_number_severity_warning(self):
        statement = make_statement([txn()], account_number=None)
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])

        result = validate_statement(statement, layout, report)
        for issue in result.issues:
            if issue.issue_type == "missing_account_number":
                assert issue.severity == "WARNING"

    def test_present_account_number_no_warning(self):
        statement = make_statement([txn()], account_number="9876543210")
        layout = make_layout()
        report = make_report([make_assignment_result(txn())])

        result = validate_statement(statement, layout, report)
        types = [i.issue_type for i in result.issues]
        assert "missing_account_number" not in types


# ===========================================================================
# Pass / fail determination
# ===========================================================================

class TestPassFail:
    def test_no_errors_passes(self):
        statement = make_statement([txn()], account_number="12345")
        layout = make_layout(detection_method="header")
        report = make_report([make_assignment_result(txn(), raw_amounts=["1,000.00"])])

        result = validate_statement(statement, layout, report)
        assert result.passed is True

    def test_error_severity_causes_failure(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="none")
        report = make_report([make_assignment_result(txn())])

        result = validate_statement(statement, layout, report)
        assert result.passed is False

    def test_only_warnings_still_passes(self):
        # data_inference adds WARNING, missing account adds WARNING — no ERRORs
        statement = make_statement([txn()], account_number=None)
        layout = make_layout(detection_method="data_inference")
        report = make_report([make_assignment_result(txn(), raw_amounts=["1,000.00"])])

        result = validate_statement(statement, layout, report)
        assert result.passed is True


# ===========================================================================
# Confidence scoring
# ===========================================================================

class TestConfidenceScoring:
    def test_perfect_case_high_confidence(self):
        txns = [txn(amount=float(i * 100)) for i in range(1, 6)]
        statement = make_statement(txns, account_number="12345")
        layout = make_layout(detection_method="header", has_amount_col=True)
        results = [make_assignment_result(t, raw_amounts=["100.00"], source_row_index=i)
                   for i, t in enumerate(txns)]
        report = make_report(results)

        result = validate_statement(statement, layout, report)
        assert result.confidence > 0.5

    def test_no_layout_gives_zero_confidence(self):
        statement = make_statement([txn()])
        layout = make_layout(detection_method="none")
        report = make_report([make_assignment_result(txn())])

        result = validate_statement(statement, layout, report)
        assert result.confidence == 0.0

    def test_data_inference_lower_than_header(self):
        txns = [txn(amount=1000.0)]
        statement = make_statement(txns, account_number="12345")

        layout_header = make_layout(detection_method="header")
        layout_infer = make_layout(detection_method="data_inference")

        report = make_report([make_assignment_result(txn(), raw_amounts=["1,000.00"])])

        conf_header = validate_statement(statement, layout_header, report).confidence
        conf_infer = validate_statement(statement, layout_infer, report).confidence
        assert conf_header >= conf_infer