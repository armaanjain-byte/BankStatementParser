"""
Unit tests for src.transaction_reconstructor.

Pipeline stage: ROW RECONSTRUCTION (OCR tokens -> logical table rows).

Function under test: reconstruct_rows(tokens, threshold=None)
  Expected input:  List[OCRToken], each with .text, .confidence, .bbox
  Expected output: List[Row], top-to-bottom, tokens sorted left-to-right,
                   row.text == row.original_text immediately after build.

Edge cases covered:
  - empty token list
  - single token
  - tokens that are vertically close (same row) vs far apart (different rows)
  - adaptive threshold computation (no explicit threshold given)
  - explicit threshold override
  - tokens out of reading order (must be re-sorted by x within a row)

Failure cases covered:
  - Row.original_text must be IMMUTABLE relative to the initial text
    (mutating row.text later must not change original_text) — this is
    the baseline that pipeline_analyzer.py's recovery detection depends on.
"""

from __future__ import annotations

import pytest

from src.transaction_reconstructor import Row, reconstruct_rows
from tests.fixtures.ocr_tokens import make_token


@pytest.mark.unit
class TestReconstructRows:
    def test_empty_token_list_returns_empty_rows(self):
        """WHY: guards against IndexError on sorted_tokens[0] for empty input.
        BUG PREVENTED: a regression where reconstruct_rows([]) crashes instead
        of returning []. main.py calls this on every page; an empty-page OCR
        result must not crash the whole batch run."""
        assert reconstruct_rows([]) == []

    def test_single_token_produces_single_row(self):
        """WHY: smallest non-trivial case — one token must become one row
        containing exactly that token's text."""
        token = make_token("Hello", bbox=(0, 0, 50, 20))
        rows = reconstruct_rows([token])
        assert len(rows) == 1
        assert rows[0].text == "Hello"
        assert rows[0].token_count == 1

    def test_tokens_with_same_y_center_form_one_row(self):
        """WHY: the core clustering behavior — tokens on the same visual line
        (similar y_center) must be grouped into a single Row, regardless of
        the order they're passed in.
        BUG PREVENTED: tokens on one line being split into separate rows,
        which would break date/amount co-occurrence in transaction_parser."""
        tokens = [
            make_token("12/03/2021", bbox=(0, 100, 80, 120)),
            make_token("UPI", bbox=(90, 102, 120, 122)),
            make_token("226.14", bbox=(400, 98, 460, 118)),
        ]
        rows = reconstruct_rows(tokens)
        assert len(rows) == 1
        assert rows[0].text == "12/03/2021 UPI 226.14"

    def test_tokens_with_different_y_center_form_separate_rows(self):
        """WHY: tokens on visually distinct lines must NOT be merged into one
        row, or transaction_parser would concatenate unrelated transactions."""
        tokens = [
            make_token("Row1", bbox=(0, 0, 40, 20)),
            make_token("Row2", bbox=(0, 100, 40, 120)),
        ]
        rows = reconstruct_rows(tokens, threshold=5.0)
        assert len(rows) == 2
        assert rows[0].text == "Row1"
        assert rows[1].text == "Row2"

    def test_rows_ordered_top_to_bottom(self):
        """WHY: downstream code (date interpolation in date_normalizer,
        header-zone detection in header_extractor) assumes rows are in
        top-to-bottom reading order."""
        tokens = [
            make_token("Bottom", bbox=(0, 200, 60, 220)),
            make_token("Top", bbox=(0, 0, 60, 20)),
            make_token("Middle", bbox=(0, 100, 60, 120)),
        ]
        rows = reconstruct_rows(tokens, threshold=5.0)
        assert [r.text for r in rows] == ["Top", "Middle", "Bottom"]

    def test_tokens_within_row_sorted_left_to_right(self):
        """WHY: token order within a row must reflect reading order (x_min
        ascending) even if input tokens are given right-to-left.
        BUG PREVENTED: scrambled row text like "226.14 UPI 12/03/2021"
        which would break DATE_RE/MONEY_RE position-dependent logic."""
        tokens = [
            make_token("226.14", bbox=(400, 100, 460, 120)),
            make_token("12/03/2021", bbox=(0, 100, 80, 120)),
            make_token("UPI", bbox=(90, 100, 120, 120)),
        ]
        rows = reconstruct_rows(tokens, threshold=5.0)
        assert rows[0].text == "12/03/2021 UPI 226.14"

    def test_adaptive_threshold_uses_median_token_height(self):
        """WHY: when threshold is None, the function computes
        threshold = median(token_height) * 0.5. This test pins that
        behavior so a future refactor can't silently change the multiplier
        without a visible test failure."""
        # Token height = 20px -> adaptive threshold = 10px.
        # Two tokens 8px apart in y_center should merge; one 30px apart
        # (with the same baseline tokens) should not.
        close_tokens = [
            make_token("A", bbox=(0, 0, 20, 20)),    # y_center=10
            make_token("B", bbox=(25, 8, 45, 28)),   # y_center=18, diff=8 <= 10
        ]
        rows = reconstruct_rows(close_tokens)
        assert len(rows) == 1

        far_tokens = [
            make_token("A", bbox=(0, 0, 20, 20)),     # y_center=10
            make_token("B", bbox=(25, 40, 45, 60)),   # y_center=50, diff=40 > 10
        ]
        rows = reconstruct_rows(far_tokens)
        assert len(rows) == 2

    def test_explicit_threshold_overrides_adaptive(self):
        """WHY: callers (e.g. CLI in __main__) may pass an explicit threshold;
        it must take precedence over the adaptive computation."""
        tokens = [
            make_token("A", bbox=(0, 0, 20, 20)),     # y_center=10
            make_token("B", bbox=(25, 40, 45, 60)),   # y_center=50, diff=40
        ]
        # With a huge explicit threshold, these should merge despite being
        # far apart (adaptive would have kept them separate).
        rows = reconstruct_rows(tokens, threshold=100.0)
        assert len(rows) == 1

    def test_original_text_is_immutable_baseline(self):
        """WHY: pipeline_analyzer.analyze_row_recovery() and evaluation.py's
        recovered_date_parse_count rely on row.original_text staying equal
        to the pre-normalization text even after row.text is mutated by
        normalize_date_strings/normalize_amount_strings.
        BUG PREVENTED: a regression where original_text is a reference to
        the same mutable string/object as text (it isn't, since str is
        immutable in Python, but this test guards the *intent* even if Row
        is refactored to use a mutable container)."""
        token = make_token("23/12/2019", bbox=(0, 0, 80, 20))
        rows = reconstruct_rows([token])
        row = rows[0]
        assert row.original_text == "23/12/2019"

        # Simulate a downstream normalizer mutating row.text
        row.text = "2019-12-23"

        assert row.original_text == "23/12/2019", (
            "original_text must remain the pre-normalization baseline"
        )
        assert row.text == "2019-12-23"

    def test_row_x_min_x_max_avg_confidence_properties(self):
        """WHY: layout_detector and semantic_assigner read x_min/x_max/
        avg_confidence off Row to do geometric matching. These derived
        properties must reflect the constituent tokens correctly."""
        tokens = [
            make_token("A", confidence=0.8, bbox=(10, 0, 30, 20)),
            make_token("B", confidence=0.6, bbox=(50, 0, 90, 20)),
        ]
        rows = reconstruct_rows(tokens, threshold=5.0)
        row = rows[0]
        assert row.x_min == 10
        assert row.x_max == 90
        assert row.avg_confidence == pytest.approx(0.7)

    def test_empty_row_properties_do_not_crash(self):
        """WHY: Row() can be constructed with no tokens (e.g. in
        transaction_parser.__main__ rehydration, which builds
        Row(y_center=..., tokens=[], text=...)). x_min/x_max/avg_confidence
        must handle the empty-tokens case gracefully (return 0 / 0.0)
        rather than raising on min()/max() of an empty sequence."""
        row = Row(y_center=0.0, tokens=[], text="dummy", original_text="dummy")
        assert row.x_min == 0
        assert row.x_max == 0
        assert row.avg_confidence == 0.0
        assert row.token_count == 0