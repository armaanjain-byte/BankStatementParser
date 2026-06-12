"""
tests/unit/test_header_extractor.py — Unit tests for header_extractor.py

Tests cover:
  - Bank name extraction (exact match, OCR mis-reads, abbreviations)
  - Account number extraction (labeled, statement-of-account, masked)
  - Account holder extraction (labeled, heuristic)
  - Statement period extraction (explicit, from-to, generic range)
  - Currency detection (symbol, ISO code, OCR mis-reads)
  - Full extract_metadata pipeline (two-pass header + fallback)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import make_token
from src.header_extractor import (
    ExtractionResult,
    _extract_bank_name,
    _extract_account_number,
    _extract_account_holder,
    _extract_currency,
    _extract_statement_period,
    _header_tokens,
    extract_metadata,
)
from src.schema import OCRToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokens_from_texts(*texts: str, y: int = 10) -> list[OCRToken]:
    """Build a row of header tokens all sitting at y=10."""
    return [make_token(t, x=i * 120, y=y) for i, t in enumerate(texts)]


def tokens_with_y(items: list[tuple[str, int]]) -> list[OCRToken]:
    """Build tokens at varying y positions for header-zone tests."""
    return [make_token(text, x=0, y=y) for text, y in items]


# ===========================================================================
# _header_tokens
# ===========================================================================

class TestHeaderTokens:
    def test_returns_top_fraction_only(self):
        tokens = tokens_with_y([("header", 10), ("middle", 500), ("footer", 990)])
        result = _header_tokens(tokens, fraction=0.35)
        texts = [t.text for t in result]
        assert "header" in texts
        assert "footer" not in texts

    def test_empty_list_returns_empty(self):
        assert _header_tokens([]) == []

    def test_all_tokens_in_header_zone(self):
        tokens = tokens_with_y([("a", 5), ("b", 10)])
        assert len(_header_tokens(tokens, fraction=0.5)) == 2

    def test_fraction_boundary_inclusive(self):
        # A token whose y_min is exactly at cutoff should be included.
        tokens = tokens_with_y([("t", 350)])  # max_y=350, cutoff=350*0.35=122.5
        result = _header_tokens(tokens, fraction=1.0)
        assert len(result) == 1


# ===========================================================================
# _extract_bank_name
# ===========================================================================

class TestExtractBankName:
    def test_hdfc_exact(self):
        tokens = tokens_from_texts("HDFC", "Bank", "Statement")
        name, conf = _extract_bank_name(tokens)
        assert name == "HDFC Bank"
        assert conf > 0.0

    def test_sbi_abbreviation(self):
        tokens = tokens_from_texts("SBI", "Account", "Details")
        name, conf = _extract_bank_name(tokens)
        assert name == "State Bank of India"

    def test_icici_lowercase_ocr(self):
        tokens = tokens_from_texts("icici bank statement")
        name, conf = _extract_bank_name(tokens)
        assert name == "ICICI Bank"

    def test_kotak_partial(self):
        tokens = tokens_from_texts("Kotak", "Mahindra", "Bank")
        name, conf = _extract_bank_name(tokens)
        assert "Kotak" in name

    def test_unknown_bank_returns_none(self):
        tokens = tokens_from_texts("Random Corp Ltd")
        name, conf = _extract_bank_name(tokens)
        assert name is None
        assert conf == 0.0

    def test_empty_tokens(self):
        name, conf = _extract_bank_name([])
        assert name is None
        assert conf == 0.0

    def test_axis_bank_detected(self):
        tokens = tokens_from_texts("Axis", "Bank", "2021")
        name, conf = _extract_bank_name(tokens)
        assert name == "Axis Bank"

    def test_international_bank_hsbc(self):
        tokens = tokens_from_texts("HSBC", "Statement", "of", "Account")
        name, conf = _extract_bank_name(tokens)
        assert name == "HSBC"


# ===========================================================================
# _extract_account_number
# ===========================================================================

class TestExtractAccountNumber:
    def test_labeled_account_number(self):
        tokens = tokens_from_texts("Account", "Number:", "910010041701828")
        acct, conf = _extract_account_number(tokens)
        assert acct == "910010041701828"
        assert conf > 0.0

    def test_account_no_colon(self):
        tokens = tokens_from_texts("Account No: 201301401")
        acct, conf = _extract_account_number(tokens)
        assert acct is not None
        assert "201301401" in acct

    def test_masked_account_number(self):
        tokens = tokens_from_texts("A/C No: XXXX1234")
        acct, conf = _extract_account_number(tokens)
        assert acct is not None

    def test_statement_of_account(self):
        tokens = tokens_from_texts("Statement of Account: 2518XXXXXX0946")
        acct, conf = _extract_account_number(tokens)
        assert acct is not None

    def test_no_account_number_returns_none(self):
        tokens = tokens_from_texts("HDFC Bank", "Monthly Statement")
        acct, conf = _extract_account_number(tokens)
        assert acct is None
        assert conf == 0.0

    def test_short_account_rejected(self):
        # Account numbers shorter than 4 chars should not be extracted
        tokens = tokens_from_texts("Account No: 12")
        acct, conf = _extract_account_number(tokens)
        # Either None or must be at least 4 chars
        if acct is not None:
            assert len(acct) >= 4


# ===========================================================================
# _extract_account_holder
# ===========================================================================

class TestExtractAccountHolder:
    def test_customer_name_label(self):
        tokens = tokens_from_texts("Customer Name: John Smith")
        holder, conf = _extract_account_holder(tokens)
        assert holder is not None
        assert "John" in holder or "Smith" in holder

    def test_account_holder_label(self):
        tokens = tokens_from_texts("Account Holder: Jane Doe")
        holder, conf = _extract_account_holder(tokens)
        assert holder is not None

    def test_heuristic_two_word_name(self):
        # Two alpha words that aren't bank keywords should match heuristic
        tokens = tokens_from_texts("Ramesh Kumar")
        holder, conf = _extract_account_holder(tokens)
        assert holder == "Ramesh Kumar"
        assert conf > 0.0

    def test_skip_keyword_tokens(self):
        # "HDFC Bank" should not match as an account holder
        tokens = tokens_from_texts("HDFC Bank")
        holder, conf = _extract_account_holder(tokens)
        assert holder is None or ("HDFC" not in holder and "Bank" not in holder)

    def test_empty_tokens_returns_none(self):
        holder, conf = _extract_account_holder([])
        assert holder is None
        assert conf == 0.0


# ===========================================================================
# _extract_statement_period
# ===========================================================================

class TestExtractStatementPeriod:
    def test_explicit_period_label(self):
        tokens = tokens_from_texts(
            "For Period: 01/09/2019 to 30/09/2019"
        )
        frm, to, conf = _extract_statement_period(tokens)
        assert frm == "01/09/2019"
        assert to == "30/09/2019"
        assert conf > 0.0

    def test_from_date_to_date(self):
        tokens = tokens_from_texts(
            "(From 26/09/2019  25/12/2019)"
        )
        frm, to, conf = _extract_statement_period(tokens)
        assert frm is not None
        assert to is not None

    def test_generic_date_range(self):
        tokens = tokens_from_texts("01/01/2021 to 31/01/2021")
        frm, to, conf = _extract_statement_period(tokens)
        assert frm == "01/01/2021"
        assert to == "31/01/2021"
        assert conf > 0.0

    def test_no_period_returns_nones(self):
        tokens = tokens_from_texts("HDFC Bank", "Account Summary")
        frm, to, conf = _extract_statement_period(tokens)
        assert frm is None
        assert to is None
        assert conf == 0.0

    def test_low_confidence_for_generic_range(self):
        # Generic range (no label) should have lower confidence than labeled period
        tokens = tokens_from_texts("01/01/2021 to 31/01/2021")
        _, _, conf_generic = _extract_statement_period(tokens)
        tokens2 = tokens_from_texts("For Period: 01/01/2021 to 31/01/2021")
        _, _, conf_labeled = _extract_statement_period(tokens2)
        assert conf_generic <= conf_labeled


# ===========================================================================
# _extract_currency
# ===========================================================================

class TestExtractCurrency:
    def test_rupee_symbol(self):
        tokens = tokens_from_texts("₹", "10,000.00")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "INR"

    def test_rs_abbreviation(self):
        tokens = tokens_from_texts("Rs.", "5,000.00")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "INR"

    def test_inr_iso_code(self):
        tokens = tokens_from_texts("Currency", "INR")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "INR"

    def test_dollar_symbol(self):
        tokens = tokens_from_texts("$", "1,234.56")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "USD"

    def test_explicit_currency_label(self):
        tokens = tokens_from_texts("Currency: USD")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "USD"

    def test_pound_symbol(self):
        tokens = tokens_from_texts("£100.00", "balance")
        code, conf = _extract_currency(tokens, tokens)
        assert code == "GBP"

    def test_no_currency_returns_none(self):
        tokens = tokens_from_texts("Account", "Statement")
        code, conf = _extract_currency(tokens, tokens)
        assert code is None
        assert conf == 0.0

    def test_fallback_to_all_tokens(self):
        # Currency symbol only in body, not in header zone
        header = tokens_from_texts("Account", "Statement")
        all_tokens = header + tokens_from_texts("₹", "Opening", "Balance")
        code, conf = _extract_currency(header, all_tokens)
        assert code == "INR"


# ===========================================================================
# extract_metadata — full pipeline
# ===========================================================================

class TestExtractMetadata:
    def _make_statement_tokens(self) -> list[OCRToken]:
        """Synthetic bank statement header."""
        return [
            make_token("HDFC Bank", x=0, y=10),
            make_token("Account Number: 12345678901", x=0, y=40),
            make_token("Customer Name: Rahul Gupta", x=0, y=70),
            make_token("For Period: 01/01/2021 to 31/01/2021", x=0, y=100),
            make_token("Currency INR", x=0, y=130),
            # Body tokens far down the page
            make_token("01/01/2021 ATM withdrawal 5,000.00", x=0, y=900),
        ]

    def test_bank_name_extracted(self):
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        assert result.metadata.bank_name == "HDFC Bank"

    def test_account_number_extracted(self):
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        assert result.metadata.account_number is not None
        assert "12345678901" in result.metadata.account_number

    def test_currency_extracted(self):
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        assert result.metadata.currency == "INR"

    def test_empty_tokens_returns_defaults(self):
        result = extract_metadata([])
        assert result.metadata.bank_name is None
        assert result.metadata.account_number is None
        assert result.confidence["bank_name"] == 0.0

    def test_result_is_extraction_result_type(self):
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        assert isinstance(result, ExtractionResult)

    def test_confidence_keys_present(self):
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        for key in ("bank_name", "account_number", "account_holder",
                    "statement_period_from", "statement_period_to", "currency"):
            assert key in result.confidence

    def test_fallback_finds_bank_outside_header(self):
        # Bank name is below the 35% header zone — should still be found via fallback
        tokens = [
            make_token("Monthly Summary", x=0, y=10),
            make_token("Account 999", x=0, y=50),
            make_token("ICICI Bank Statement", x=0, y=600),  # below header zone
        ]
        result = extract_metadata(tokens)
        assert result.metadata.bank_name == "ICICI Bank"

    def test_to_dict_is_serialisable(self):
        import json
        tokens = self._make_statement_tokens()
        result = extract_metadata(tokens)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)