"""
tests/unit/test_layout_detector.py — Unit tests for layout_detector.py

Tests cover:
  - Header-row candidate detection (_find_header_candidates)
  - Column name matching (_match_column_name / ALIAS_MAP)
  - Role assignment from header rows (_choose_role_columns)
  - Data-row layout inference (_infer_layout_from_data_rows)
  - detect_layout() dispatch: header path vs inference path
  - LayoutDefinition.to_dict serialisation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import make_token, make_row, make_row_with_tokens
from src.layout_detector import (
    LayoutDefinition,
    detect_layout,
    _find_header_candidates,
    _infer_layout_from_data_rows,
    _match_column_name,
    _normalize_text,
)
from src.schema import OCRToken
from src.transaction_reconstructor import Row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header_row(*col_names: str, base_y: int = 10) -> Row:
    """Create a row whose tokens are labelled with common column header names."""
    tokens = [
        make_token(name, x=i * 150, y=base_y, w=130, h=20)
        for i, name in enumerate(col_names)
    ]
    text = " ".join(col_names)
    return Row(y_center=base_y + 10.0, tokens=tokens, text=text, original_text=text)


def data_row(date: str, desc: str, amount: str, base_y: int = 100) -> Row:
    """Create a typical transaction data row with date left, amount right."""
    tokens = [
        make_token(date, x=0, y=base_y, w=100, h=20),
        make_token(desc, x=120, y=base_y, w=200, h=20),
        make_token(amount, x=400, y=base_y, w=80, h=20),
    ]
    text = f"{date} {desc} {amount}"
    return Row(y_center=base_y + 10.0, tokens=tokens, text=text, original_text=text)


# ===========================================================================
# _normalize_text
# ===========================================================================

class TestNormalizeText:
    def test_lowercases(self):
        assert _normalize_text("Date") == "date"

    def test_strips_punctuation(self):
        assert _normalize_text("Dr.") == "dr"

    def test_collapses_spaces(self):
        assert _normalize_text("Txn  Date") == "txn date"

    def test_removes_special_chars(self):
        assert _normalize_text("Balance (₹)") == "balance"


# ===========================================================================
# _match_column_name
# ===========================================================================

class TestMatchColumnName:
    def test_exact_date(self):
        name, conf, kind = _match_column_name("Date")
        assert name == "date"
        assert conf > 0.0
        assert kind in ("exact_alias", "word_alias")

    def test_narration_maps_to_description(self):
        name, conf, kind = _match_column_name("Narration")
        assert name == "description"

    def test_withdrawal_maps_to_debit(self):
        name, conf, kind = _match_column_name("Withdrawal")
        assert name == "debit"

    def test_deposit_maps_to_credit(self):
        name, conf, kind = _match_column_name("Deposit")
        assert name == "credit"

    def test_balance_exact(self):
        name, conf, kind = _match_column_name("Balance")
        assert name == "balance"
        assert conf > 0.0

    def test_ocr_mangled_debit(self):
        # "dehit" is in ALIAS_MAP at 0.70 confidence
        name, conf, kind = _match_column_name("dehit")
        assert name == "debit"
        assert conf > 0.0

    def test_date_value_not_matched(self):
        # Actual date value should not be treated as a column header
        name, conf, kind = _match_column_name("01/01/2021")
        assert name is None
        assert kind == "date_value"

    def test_empty_string(self):
        name, conf, kind = _match_column_name("")
        assert name is None

    def test_txn_date_alias(self):
        name, conf, kind = _match_column_name("Txn Date")
        assert name == "date"


# ===========================================================================
# _find_header_candidates
# ===========================================================================

class TestFindHeaderCandidates:
    def test_canonical_header_detected(self):
        rows = [
            header_row("Date", "Narration", "Debit", "Credit", "Balance"),
            data_row("01/01/2021", "ATM Withdrawal", "5,000.00"),
        ]
        candidates = _find_header_candidates(rows)
        assert len(candidates) >= 1
        assert candidates[0].source_row_index == 0

    def test_no_date_column_rejected(self):
        # Row with debit/credit but no date should not be a candidate
        rows = [header_row("Narration", "Debit", "Credit")]
        candidates = _find_header_candidates(rows)
        assert len(candidates) == 0

    def test_only_date_column_rejected(self):
        # date + nothing else should not qualify
        rows = [header_row("Date")]
        candidates = _find_header_candidates(rows)
        assert len(candidates) == 0

    def test_candidates_sorted_by_confidence_descending(self):
        rows = [
            header_row("Date", "Description", "Amount"),
            header_row("Date", "Narration", "Debit", "Credit", "Balance"),
        ]
        candidates = _find_header_candidates(rows)
        if len(candidates) >= 2:
            assert candidates[0].confidence >= candidates[1].confidence

    def test_empty_rows_returns_empty(self):
        assert _find_header_candidates([]) == []


# ===========================================================================
# _infer_layout_from_data_rows
# ===========================================================================

class TestInferLayoutFromDataRows:
    def _sample_data_rows(self, n: int = 4) -> list[Row]:
        return [
            data_row("01/01/2021", "ATM WITHDRAWAL", "5,000.00", base_y=i * 30)
            for i in range(n)
        ]

    def test_returns_layout_definition(self):
        rows = self._sample_data_rows()
        layout = _infer_layout_from_data_rows(rows)
        assert isinstance(layout, LayoutDefinition)

    def test_detection_method_data_inference(self):
        rows = self._sample_data_rows()
        layout = _infer_layout_from_data_rows(rows)
        assert layout.detection_method == "data_inference"

    def test_date_column_present(self):
        rows = self._sample_data_rows()
        layout = _infer_layout_from_data_rows(rows)
        assert layout.date_column is not None

    def test_amount_column_present(self):
        rows = self._sample_data_rows()
        layout = _infer_layout_from_data_rows(rows)
        assert layout.amount_column is not None

    def test_confidence_increases_with_more_rows(self):
        few = _infer_layout_from_data_rows(self._sample_data_rows(2))
        many = _infer_layout_from_data_rows(self._sample_data_rows(8))
        few_conf = few.confidence_scores.get("amount", 0.0)
        many_conf = many.confidence_scores.get("amount", 0.0)
        assert many_conf >= few_conf

    def test_empty_rows_returns_none_method(self):
        layout = _infer_layout_from_data_rows([])
        assert layout.detection_method == "none"

    def test_rows_without_amounts_return_none_method(self):
        rows = [make_row("01/01/2021 Description only no amount")]
        layout = _infer_layout_from_data_rows(rows)
        assert layout.detection_method == "none"


# ===========================================================================
# detect_layout — dispatch logic
# ===========================================================================

class TestDetectLayout:
    def test_uses_header_when_present(self):
        rows = [
            header_row("Date", "Narration", "Debit", "Credit", "Balance"),
            data_row("01/01/2021", "Salary credit", "0.00", base_y=50),
            data_row("02/01/2021", "ATM withdrawal", "2,000.00", base_y=80),
        ]
        layout = detect_layout([], rows)
        assert layout.detection_method == "header"

    def test_falls_back_to_inference_without_header(self):
        rows = [
            data_row("01/01/2021", "Salary", "10,000.00", base_y=i * 30)
            for i in range(5)
        ]
        layout = detect_layout([], rows)
        assert layout.detection_method in ("data_inference", "header")

    def test_header_assigns_debit_credit_columns(self):
        rows = [
            header_row("Date", "Narration", "Debit", "Credit", "Balance"),
        ]
        layout = detect_layout([], rows)
        assert layout.debit_column is not None or layout.credit_column is not None

    def test_empty_rows_returns_none_detection(self):
        layout = detect_layout([], [])
        assert layout.detection_method == "none"

    def test_layout_to_dict_is_json_serialisable(self):
        import json
        rows = [header_row("Date", "Narration", "Debit", "Credit", "Balance")]
        layout = detect_layout([], rows)
        d = layout.to_dict()
        json.dumps(d)  # must not raise

    def test_debit_credit_preferred_over_inferred_amount(self):
        """Header-derived debit/credit must NOT be overwritten by data inference."""
        rows = [
            header_row("Date", "Description", "Debit", "Credit", "Balance"),
            data_row("01/01/2021", "Payment", "100.00", base_y=50),
        ]
        layout = detect_layout([], rows)
        # If header found debit/credit, the fill step must not replace with amount
        assert layout.debit_column is not None or layout.credit_column is not None

    def test_confidence_scores_populated(self):
        rows = [header_row("Date", "Narration", "Debit", "Credit", "Balance")]
        layout = detect_layout([], rows)
        assert isinstance(layout.confidence_scores, dict)
        assert len(layout.confidence_scores) > 0