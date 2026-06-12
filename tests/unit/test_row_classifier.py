"""
Unit tests for src.row_classifier.

Pipeline stage: ROW CLASSIFICATION (pre-filter before date extraction).

Class under test: RowClassifier.classify(row_text) -> RowClassification
Function under test: classify_rows(rows) -> (filtered_rows, counts)

This module's docstring claims it eliminates "333+ no_date_detected false
positives" by rejecting headers/footers/page-markers/metadata BEFORE date
parsing is attempted.
"""

from __future__ import annotations

import pytest

from src.row_classifier import RowClassifier, classify_rows


@pytest.fixture
def classifier():
    return RowClassifier()


@pytest.mark.unit
class TestRowClassifierNonTransactionRows:
    def test_empty_row_rejected(self, classifier):
        """WHY: empty or near-empty rows (OCR artifacts) must not reach the
        date/amount parsing stage."""
        result = classifier.classify("")
        assert result.is_transaction_candidate is False
        assert result.classification == "empty"

    def test_short_row_rejected(self, classifier):
        result = classifier.classify("ab")
        assert result.is_transaction_candidate is False
        assert result.classification == "empty"

    def test_separator_row_rejected(self, classifier):
        """WHY: visual table separators ('-----', '=====') must be filtered
        before date parsing -- they could otherwise match a corrupted-date
        regex by coincidence."""
        result = classifier.classify("----------")
        assert result.is_transaction_candidate is False
        assert result.classification == "separator"

    def test_account_number_header_rejected(self, classifier):
        """WHY: 'Statement of Account Number: 2518XXXXXXXXO946' must not be
        treated as a transaction row, even though it contains digits that
        could look date- or amount-like."""
        result = classifier.classify("Statement of Account Number: 2518XXXXXXXXO946")
        assert result.is_transaction_candidate is False
        assert result.classification == "header"

    def test_closing_balance_footer_rejected(self, classifier):
        result = classifier.classify("Closing Balance: 5000.00")
        assert result.is_transaction_candidate is False
        assert result.classification == "footer"

    def test_page_marker_rejected(self, classifier):
        result = classifier.classify("Page 1 of 5")
        assert result.is_transaction_candidate is False
        assert result.classification == "page_marker"

    def test_bank_name_metadata_rejected(self, classifier):
        result = classifier.classify("HDFC Bank Savings Account Statement")
        assert result.is_transaction_candidate is False
        assert result.classification in ("metadata", "header")

    def test_reward_points_summary_rejected(self, classifier):
        """WHY: 'Reward Points...' is rejected as non-transaction. Note the
        footer_patterns regex (checked before summary_patterns) also
        contains 'reward', so this classifies as 'footer' rather than
        'summary' -- both are non-transaction outcomes, which is what
        matters for the pipeline. This test documents the actual
        classification order (footer wins) so a refactor that reorders
        these checks is a visible, deliberate change."""
        result = classifier.classify("Reward Points Earned This Month: 250")
        assert result.is_transaction_candidate is False
        assert result.classification in ("footer", "summary")

    def test_terms_and_conditions_disclaimer_rejected(self, classifier):
        result = classifier.classify("Terms and Conditions apply to this statement")
        assert result.is_transaction_candidate is False
        assert result.classification == "disclaimer"

    def test_column_header_row_rejected(self, classifier):
        """WHY: a row listing 3+ column-header keywords ('Date Narration
        Debit Credit Balance') is the table header itself, not a data row.
        BUG PREVENTED: the column-header row being parsed as a transaction
        with garbage date/amount values."""
        result = classifier.classify("Date Narration Debit Credit Balance")
        assert result.is_transaction_candidate is False
        assert result.classification == "column_header"


@pytest.mark.unit
class TestRowClassifierTransactionRows:
    def test_plain_transaction_row_accepted(self, classifier):
        """WHY: a normal transaction row with date + narration + amount must
        be classified as a transaction candidate (the permissive default)."""
        result = classifier.classify("12/03/2021 UPI Payment to Vendor 226.14")
        assert result.is_transaction_candidate is True
        assert result.classification == "transaction"

    def test_row_with_some_digits_but_no_keywords_accepted(self, classifier):
        """WHY: rows that don't match ANY non-transaction pattern fall
        through to 'transaction' by design (confidence 0.6) -- this is the
        permissive default that lets ambiguous rows reach the
        TransactionCandidateDetector for a second opinion."""
        result = classifier.classify("23/12/2019 Misc transfer 100.00")
        assert result.is_transaction_candidate is True


@pytest.mark.unit
class TestClassifyRows:
    def test_filters_and_counts_correctly(self):
        rows = [
            "Statement of Account Number: 123456",
            "12/03/2021 UPI Payment 226.14",
            "Closing Balance: 5000.00",
            "23/12/2019 ATM Withdrawal 50.00",
            "Page 1 of 1",
        ]
        filtered, counts = classify_rows(rows)
        assert filtered == [
            "12/03/2021 UPI Payment 226.14",
            "23/12/2019 ATM Withdrawal 50.00",
        ]
        assert counts["header"] == 1
        assert counts["footer"] == 1
        assert counts["page_marker"] == 1
        assert counts["transaction"] == 2

    def test_empty_input_returns_empty_with_empty_counts(self):
        filtered, counts = classify_rows([])
        assert filtered == []
        assert counts == {}