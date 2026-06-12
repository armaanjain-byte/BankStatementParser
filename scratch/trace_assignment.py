"""
scratch/trace_assignment.py — Assignment failure tracer for target statements.

Runs the full pipeline for each target statement using cached OCR and
traces exactly why assigned_transaction_count = 0.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, ".")

from pathlib import Path
from src.layout_detector import load_ocr_tokens, detect_layout
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.transaction_parser import parse_rows
from src.semantic_assigner import assign_rows, build_semantic_assignment_report

# Statements with parsed_rows > 0 but assigned_transactions = 0
TARGETS = ["11", "13", "16", "66", "90", "92", "97"]

for sid in TARGETS:
    ocr_cache = Path(f"outputs/{sid}_ocr.json")
    if not ocr_cache.exists():
        # Try outputs directory only
        print(f"\n{'='*80}")
        print(f"STATEMENT {sid}  — OCR cache NOT FOUND at {ocr_cache}, skipping")
        continue

    print(f"\n{'='*80}")
    print(f"STATEMENT {sid}")
    print(f"{'='*80}")

    tokens = load_ocr_tokens(ocr_cache)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    parsed = parse_rows(rows)

    print(f"  tokens={len(tokens)}  rows={len(rows)}  parsed={len(parsed)}")
    print(f"  layout.detection_method = {layout.detection_method!r}")
    print(f"  layout.confidence_scores = {layout.confidence_scores}")

    # Show money columns
    from src.semantic_assigner import _layout_money_columns
    money_cols = _layout_money_columns(layout)
    print(f"  money_columns = {[(r, c.x_center) for r,c in money_cols]}")

    report = build_semantic_assignment_report(parsed, rows, layout, statement_id=sid)
    print(f"  assignment_count={report.assignment_count}  assigned_row_count={report.assigned_row_count}")

    # Trace each parsed row's assignment result
    print()
    print(f"  PARSED ROW ASSIGNMENTS:")
    for ar in report.rows:
        amt_none = all(v is None for v in ar.assigned_values.values())
        print(f"    row_idx={ar.source_row_index}  amounts={ar.raw_amounts}")
        print(f"      reason   = {ar.reason!r}")
        print(f"      values   = {ar.assigned_values}")
        print(f"      all_none = {amt_none}  conf={ar.confidence}")
        print(f"      text     = {ar.raw_text[:80]}")
        print()
