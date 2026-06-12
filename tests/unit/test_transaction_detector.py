"""
Unit tests for src.transaction_detector.

Pipeline stage: TRANSACTION CANDIDATE DETECTION (second-stage heuristic
scoring, after row_classifier's pattern-based rejection).

Class under test: TransactionCandidateDetector.detect(row_text) -> DetectionResult
  score >= 0.5 => is_transaction_candidate = True
"""

from __future__ import annotations

import pytest

from src.transaction_detector import TransactionCandidateDetector


@pytest.fixture
def detector():
    return TransactionCandidateDetector()


@pytest.mark.unit
class TestTransactionCandidateDetector:
    def test_negative_keyword_instant_rejection(self, detector):
        """WHY: rows containing negative-signal keywords ('Page',
        'Statement', 'IFSC', etc.) are instantly rejected with score 0.0,
        regardless of any positive signals also present.
        BUG PREVENTED: a header row like 'Statement Date 12/03/2021' being
        scored as a transaction because it also contains a date."""
        result = detector.detect("Statement Date 12/03/2021 Account Number 123")
        assert result.is_transaction_candidate is False
        assert result.confidence_score == 0.0
        assert result.reasons == ["Contains negative signal keywords"]

    def test_date_plus_two_amounts_plus_keyword_high_confidence(self, detector):
        """WHY: a row with a date (0.4) + 2 amounts (0.3 + 0.2) +
        transaction keyword (0.3) = 1.2, clamped to 1.0 -- maximal
        confidence transaction row."""
        result = detector.detect("12/03/2021 UPI Payment 100.00 5000.00")
        assert result.is_transaction_candidate is True
        assert result.confidence_score == pytest.approx(1.0)

    def test_date_and_single_amount_meets_threshold(self, detector):
        """WHY: date (0.4) + 1 amount (0.3) = 0.7 >= 0.5 threshold ->
        candidate. This is the minimal 'looks like a transaction' case."""
        result = detector.detect("12/03/2021 Some narration 50.00")
        assert result.is_transaction_candidate is True
        assert result.confidence_score >= 0.5

    def test_date_only_below_threshold(self, detector):
        """WHY: a date alone (0.4) is below the 0.5 threshold -- a bare date
        with no amount or transaction keyword should NOT be treated as a
        transaction candidate (it's likely a stray date in narration or a
        statement-period header)."""
        result = detector.detect("12/03/2021")
        assert result.is_transaction_candidate is False
        assert result.confidence_score == pytest.approx(0.4)

    def test_cr_dr_indicator_contributes_to_score(self, detector):
        """WHY: 'Cr'/'Dr' suffixes are a strong transaction signal (0.2)."""
        result = detector.detect("12/03/2021 Balance 5000.00 Cr")
        assert "Contains debit/credit indicators" in result.reasons

    def test_mangled_ocr_date_candidate_still_scores(self, detector):
        """WHY: the broader date_candidate_re catches OCR-mangled dates
        (e.g. '13.10.2021' with dot separators) that the strict DATE_RE
        might miss, contributing to the score even when DATE_RE itself
        doesn't match.
        BUG PREVENTED: a row with a slightly-mangled date and an amount
        being scored too low and silently dropped before parse_rows even
        gets a chance."""
        result = detector.detect("13.10.2021 Some Narration 100.00")
        assert result.is_transaction_candidate is True

    def test_empty_string_not_a_candidate(self, detector):
        result = detector.detect("")
        assert result.is_transaction_candidate is False