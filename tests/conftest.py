"""
tests/conftest.py — Shared pytest fixtures and path setup.

Ensures the project root (containing the `src` package) is importable
regardless of where pytest is invoked from, and exposes small reusable
fixtures used across unit/integration/regression/e2e tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root = parent of this tests/ directory (BankStatementParser-master)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.ocr_tokens import (  # noqa: E402
    make_row_from_text,
    make_row_from_words,
    make_token,
    make_tokens_in_rows,
)


@pytest.fixture
def token_factory():
    """Return the make_token helper for tests that want a callable fixture."""
    return make_token


@pytest.fixture
def row_factory():
    """Return the make_row_from_text helper."""
    return make_row_from_text


@pytest.fixture
def row_words_factory():
    """Return the make_row_from_words helper (geometry-aware)."""
    return make_row_from_words


@pytest.fixture
def multi_row_tokens_factory():
    """Return the make_tokens_in_rows helper."""
    return make_tokens_in_rows


@pytest.fixture
def simple_transaction_row():
    """A clean, unambiguous transaction row: date + description + amount."""
    return make_row_from_text("12/03/2021 UPI Payment to Vendor 226.14")


@pytest.fixture
def header_row_text():
    """A typical statement header row that must NOT be treated as a transaction."""
    return "Statement of Account Number: 2518XXXXXXXXO946"