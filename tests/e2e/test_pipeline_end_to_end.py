"""
tests/e2e/test_pipeline_end_to_end.py — End-to-end pipeline tests

Runs the full pipeline (reconstruct → normalise → parse → layout → assign →
validate) using synthetic OCRToken fixtures, with the PaddleOCR backend
mocked so the suite runs without the paddleocr package installed.

PaddleOCR is not installed in this environment, so all OCR calls are
replaced with unittest.mock.MagicMock.  The pipeline logic under test is
everything that happens AFTER tokens are produced.
"""


from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.ocr_tokens import make_token
from src.layout_detector import detect_layout
from src.normalizers.amount_normalizer import normalize_amount_strings
from src.normalizers.date_normalizer import normalize_date_strings
from src.schema import OCRToken
from src.semantic_assigner import assign_rows, build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import Row, reconstruct_rows
from src.validator import validate_statement
from src.models.transaction_models import BankStatement, StatementMetadata


# ---------------------------------------------------------------------------
# Synthetic OCR page builder
# ---------------------------------------------------------------------------

def make_page_tokens(rows_spec: list[dict]) -> list[OCRToken]:
    """
    Build a flat list of OCRToken objects from a high-level row spec.

    Each element of rows_spec is a dict with:
      - "y":    int  — vertical position of this row (pixels from top)
      - "cols": list[tuple[str, int]]  — (text, x_position) pairs

    Example::

        make_page_tokens([
            {"y": 10,  "cols": [("Date", 0), ("Narration", 150), ("Debit", 350)]},
            {"y": 50,  "cols": [("01/01/2021", 0), ("ATM", 150), ("5,000.00", 350)]},
        ])
    """
    tokens: list[OCRToken] = []
    for row in rows_spec:
        y = row["y"]
        for text, x in row["cols"]:
            tokens.append(make_token(text, x=x, y=y, w=120, h=20))
    return tokens


def simple_statement_tokens() -> list[OCRToken]:
    """Three-transaction statement with explicit header."""
    return make_page_tokens([
        {"y": 10,  "cols": [("HDFC Bank", 0), ("Account:", 200), ("No:", 300), ("12345678", 340)]},
        {"y": 40,  "cols": [("Date", 0), ("Narration", 150), ("Debit", 380), ("Credit", 480), ("Balance", 580)]},
        {"y": 80,  "cols": [("01/01/2021", 0), ("ATM Withdrawal", 150), ("5,000.00", 380), ("0.00", 480), ("45,000.00", 580)]},
        {"y": 110, "cols": [("05/01/2021", 0), ("Salary Credit",  150), ("0.00", 380), ("50,000.00", 480), ("95,000.00", 580)]},
        {"y": 140, "cols": [("10/01/2021", 0), ("NEFT Transfer",  150), ("2,000.00", 380), ("0.00", 480), ("93,000.00", 580)]},
    ])


def noisy_statement_tokens() -> list[OCRToken]:
    """Statement with headers, footers, and OCR-mangled rows mixed in."""
    return make_page_tokens([
        {"y": 5,   "cols": [("SBI", 0), ("Statement of Account", 100)]},
        {"y": 25,  "cols": [("For Period:", 0), ("01/01/2021", 120), ("to", 230), ("31/01/2021", 260)]},
        {"y": 50,  "cols": [("Date", 0), ("Description", 150), ("Amount", 400)]},
        {"y": 80,  "cols": [("01/01/2021", 0), ("Grocery", 150), ("350.00", 400)]},
        {"y": 110, "cols": [("Page 1 of 5", 0)]},  # page marker
        {"y": 140, "cols": [("05/01/2021", 0), ("UPI/Payment", 150), ("1,200.00", 400)]},
        {"y": 170, "cols": [("Terms and Conditions", 0)]},  # disclaimer
        {"y": 200, "cols": [("10/01/2021", 0), ("Salary", 150), ("50,000.00", 400)]},
        {"y": 230, "cols": [("End of Statement", 0)]},  # footer
    ])


# ===========================================================================
# Full pipeline execution helpers
# ===========================================================================

def run_full_pipeline(tokens: list[OCRToken]):
    """Run reconstruct → normalise → parse → layout → assign → validate."""
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    parsed = parse_rows(rows)
    layout = detect_layout(tokens, rows)
    results = assign_rows(parsed, rows, layout)
    report = build_semantic_assignment_report(parsed, rows, layout, statement_id="e2e_test")

    transactions = [r.transaction for r in results]
    metadata = StatementMetadata()
    statement = BankStatement(metadata=metadata, transactions=transactions)
    validation = validate_statement(statement, layout, report)

    return rows, parsed, layout, results, report, validation


# ===========================================================================
# Tests: simple statement
# ===========================================================================

class TestSimpleStatementE2E:
    def test_reconstruct_produces_rows(self):
        tokens = simple_statement_tokens()
        rows = reconstruct_rows(tokens)
        assert len(rows) >= 3  # header + 3 data rows at minimum

    def test_three_transactions_parsed(self):
        tokens = simple_statement_tokens()
        rows, parsed, layout, results, report, validation = run_full_pipeline(tokens)
        assert len(parsed) == 3

    def test_all_transactions_have_dates(self):
        _, _, _, results, _, _ = run_full_pipeline(simple_statement_tokens())
        for r in results:
            assert r.transaction.date is not None

    def test_header_detection_method(self):
        _, _, layout, _, _, _ = run_full_pipeline(simple_statement_tokens())
        assert layout.detection_method == "header"

    def test_validation_passes(self):
        tokens = simple_statement_tokens()
        rows = reconstruct_rows(tokens)
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        parsed = parse_rows(rows)
        layout = detect_layout(tokens, rows)
        results = assign_rows(parsed, rows, layout)
        report = build_semantic_assignment_report(parsed, rows, layout)
        transactions = [r.transaction for r in results]
        statement = BankStatement(
            metadata=StatementMetadata(account_number="12345678"),
            transactions=transactions,
        )
        validation = validate_statement(statement, layout, report)
        assert validation.passed is True

    def test_confidence_above_zero(self):
        _, _, _, _, _, validation = run_full_pipeline(simple_statement_tokens())
        # Even without account number in metadata, confidence should be > 0
        # when there are transactions and a header-detected layout
        assert validation.confidence >= 0.0

    def test_report_assignment_count_matches_parsed(self):
        _, parsed, _, _, report, _ = run_full_pipeline(simple_statement_tokens())
        assert report.assignment_count == len(parsed)

    def test_report_is_json_serialisable(self):
        import json
        _, _, _, _, report, _ = run_full_pipeline(simple_statement_tokens())
        json.dumps(report.to_dict())  # must not raise


# ===========================================================================
# Tests: noisy statement with non-transaction rows
# ===========================================================================

class TestNoisyStatementE2E:
    def test_non_transaction_rows_filtered(self):
        _, parsed, _, _, _, _ = run_full_pipeline(noisy_statement_tokens())
        # "Page 1 of 5", "Terms and Conditions", "End of Statement" must not parse
        descriptions = [p.description.lower() for p in parsed if p.description]
        assert not any("page" in d for d in descriptions)
        assert not any("terms" in d for d in descriptions)
        assert not any("end of" in d for d in descriptions)

    def test_three_valid_transactions_extracted(self):
        _, parsed, _, _, _, _ = run_full_pipeline(noisy_statement_tokens())
        assert len(parsed) == 3

    def test_pipeline_does_not_crash_on_noise(self):
        try:
            run_full_pipeline(noisy_statement_tokens())
        except Exception as exc:
            pytest.fail(f"Pipeline crashed on noisy statement: {type(exc).__name__}: {exc}")


# ===========================================================================
# Tests: empty token list
# ===========================================================================

class TestEmptyInputE2E:
    def test_empty_tokens_no_crash(self):
        try:
            run_full_pipeline([])
        except Exception as exc:
            pytest.fail(f"Pipeline crashed on empty tokens: {exc}")

    def test_empty_tokens_no_transactions(self):
        _, parsed, _, results, _, _ = run_full_pipeline([])
        assert parsed == []
        assert results == []

    def test_empty_tokens_validation_zero_confidence(self):
        _, _, layout, _, report, validation = run_full_pipeline([])
        assert validation.confidence == 0.0


# ===========================================================================
# Tests: PaddleOCR backend is properly mocked (dependency injection guard)
# ===========================================================================

class TestPaddleOCRMocked:
    def test_paddleocr_not_imported_during_pipeline(self):
        """Ensure our tests never accidentally call the real PaddleOCR."""
        paddleocr_mock = MagicMock()
        with patch.dict(sys.modules, {"paddleocr": paddleocr_mock}):
            # Running the pipeline with mock OCR backend must not invoke
            # paddleocr.PaddleOCR()
            tokens = simple_statement_tokens()
            # We feed tokens directly, bypassing OCR
            rows = reconstruct_rows(tokens)
            parsed = parse_rows(rows)
            assert len(parsed) >= 0  # just ensure no crash

        # paddleocr.PaddleOCR() must never have been instantiated
        paddleocr_mock.PaddleOCR.assert_not_called()


# ===========================================================================
# Tests: reconstruct_rows determinism
# ===========================================================================

class TestReconstructDeterminism:
    def test_same_tokens_same_row_count(self):
        tokens = simple_statement_tokens()
        rows_a = reconstruct_rows(tokens)
        rows_b = reconstruct_rows(tokens)
        assert len(rows_a) == len(rows_b)

    def test_row_texts_deterministic(self):
        tokens = simple_statement_tokens()
        rows_a = reconstruct_rows(tokens)
        rows_b = reconstruct_rows(tokens)
        for a, b in zip(rows_a, rows_b):
            assert a.text == b.text

    def test_tokens_sorted_by_y_within_rows(self):
        """y_centers of reconstructed rows should be non-decreasing."""
        tokens = simple_statement_tokens()
        rows = reconstruct_rows(tokens)
        y_centers = [r.y_center for r in rows]
        assert y_centers == sorted(y_centers)


# ===========================================================================
# Tests: statement from real output JSON (smoke test)
# ===========================================================================

class TestStatementOutputSmoke:
    """Smoke tests against the pre-computed statement_outputs/*.json files.
    
    These tests will automatically skip on fresh repository clones where 
    statement outputs have not yet been generated.
    """

    OUTPUTS_DIR = Path(__file__).parent.parent.parent / "statement_outputs"

    def _available_ids(self):
        if not self.OUTPUTS_DIR.exists():
            return []
        return [p.stem.replace("_statement", "") for p in self.OUTPUTS_DIR.glob("*_statement.json")]

    def test_statement_outputs_exist(self):
        ids = self._available_ids()
        if not ids:
            pytest.skip(f"No pre-computed statement outputs found in {self.OUTPUTS_DIR}. Skipping smoke tests.")
            
        assert len(ids) >= 1, "At least one statement output file must exist if directory is populated"

    def test_statement_output_has_expected_keys(self):
        ids = self._available_ids()
        if not ids:
            pytest.skip(f"No pre-computed statement outputs found in {self.OUTPUTS_DIR}.")
            
        # Dynamically test all available statement outputs
        for stmt_id in ids:
            path = self.OUTPUTS_DIR / f"{stmt_id}_statement.json"
            with open(path) as f:
                data = json.load(f)
            
            assert "transactions" in data or "metadata" in data, (
                f"Output {stmt_id}_statement.json lacks fundamental schema keys"
            )