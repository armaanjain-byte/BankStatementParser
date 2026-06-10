"""
Diagnostic script for Statement 24 semantic assignment failures.

Does NOT modify any production code. Runs the pipeline, dumps all
parsed rows, layout info, mismatch reasons, and simulates
position-based assignment.
"""
import json
import logging
import re

from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.layout_detector import detect_layout, DetectedColumn
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.constants import MONEY_RE
from src.transaction_parser import parse_rows
from src.semantic_assigner import (
    build_semantic_assignment_report,
    MONEY_ROLES,
)

logging.disable(logging.CRITICAL)

# ── Run pipeline ────────────────────────────────────────────────
tokens = extract_tokens("test_set/24.jpg", gpu=False)
rows = reconstruct_rows(tokens)
rows, _ = normalize_date_strings(rows)
rows, _ = normalize_amount_strings(rows)
layout = detect_layout(tokens, rows)
parsed = parse_rows(rows)
report = build_semantic_assignment_report(parsed, rows, layout, statement_id="24")

# ── 1. Layout information ──────────────────────────────────────
print("=" * 70)
print("SECTION 1: DETECTED LAYOUT")
print("=" * 70)
print(f"Detection Method: {layout.detection_method}")
print(f"Header Rows: {len(layout.header_rows)}")
if layout.header_rows:
    for hr in layout.header_rows:
        print(f"  Row {hr.source_row_index}: {hr.raw_text}")

money_columns = []
for role in MONEY_ROLES:
    col = getattr(layout, f"{role}_column")
    if col is not None:
        money_columns.append((role, col))
        print(f"\n  [{role.upper()} COLUMN]")
        print(f"    source_text:  {col.source_text}")
        print(f"    x_min:        {col.x_min}")
        print(f"    x_max:        {col.x_max}")
        print(f"    x_center:     {col.x_center}")
        print(f"    confidence:   {col.confidence:.4f}")

money_columns_sorted = sorted(money_columns, key=lambda x: x[1].x_center)
print(f"\nMoney columns (L->R): {[f'{r}@{c.x_center}' for r,c in money_columns_sorted]}")
print(f"Total money columns: {len(money_columns)}")

# ── 2. All 10 parsed rows ──────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: ALL 10 PARSED ROWS")
print("=" * 70)

for pr in parsed:
    row = rows[pr.source_row_index]
    print(f"\n--- ParsedRow (source_row_index={pr.source_row_index}) ---")
    print(f"  date:        {pr.date}")
    print(f"  description: {pr.description}")
    print(f"  amounts:     {pr.amounts}")
    print(f"  raw_text:    {pr.raw_text}")

    # Find x-coordinates of amount tokens in the original row
    for amt in pr.amounts:
        for token in row.tokens:
            if amt in token.text:
                x_min = token.bbox[0]
                x_max = token.bbox[2]
                x_center = (x_min + x_max) / 2
                print(f"  amount '{amt}' token x: [{x_min}, {x_max}], center={x_center:.1f}")

# ── 3. Semantic mismatch reasons ──────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: SEMANTIC MISMATCH DETAILS")
print("=" * 70)

for ar in report.rows:
    print(f"\n--- Row {ar.source_row_index} ---")
    print(f"  raw_text:      {ar.raw_text[:80]}")
    print(f"  raw_amounts:   {ar.raw_amounts}")
    print(f"  reason:        {ar.reason}")
    print(f"  expected cols: {len(money_columns)}")
    print(f"  actual amts:   {len(ar.raw_amounts)}")
    assigned_any = any(v is not None for v in ar.assigned_values.values())
    print(f"  assigned:      {assigned_any}")

# ── 4. Position-based assignment simulation ───────────────────
print("\n" + "=" * 70)
print("SECTION 4: POSITION-BASED ASSIGNMENT SIMULATION")
print("=" * 70)

recoverable = 0
total_mismatch = 0

for pr in parsed:
    row = rows[pr.source_row_index]
    ar = next(r for r in report.rows if r.source_row_index == pr.source_row_index)

    if any(v is not None for v in ar.assigned_values.values()):
        continue  # Already assigned

    total_mismatch += 1

    print(f"\n--- Row {pr.source_row_index}: {pr.amounts} ---")

    # For each amount, find its x-position from the tokens
    amount_positions = []
    for amt in pr.amounts:
        for token in row.tokens:
            if amt in token.text:
                x_center = (token.bbox[0] + token.bbox[2]) / 2
                amount_positions.append((amt, x_center))
                break

    if not amount_positions:
        print(f"  [!] Could not locate amount tokens in row geometry")
        continue

    # Calculate distance to each money column
    for amt, x_pos in amount_positions:
        print(f"\n  Amount '{amt}' at x={x_pos:.1f}")
        distances = []
        for role, col in money_columns_sorted:
            dist = abs(x_pos - col.x_center)
            distances.append((role, col.x_center, dist))
            print(f"    -> {role:>8} column (center={col.x_center:.1f}): distance={dist:.1f}px")

        nearest = min(distances, key=lambda d: d[2])
        print(f"    * Nearest: {nearest[0]} (distance={nearest[2]:.1f}px)")

    # Would this be recoverable?
    # Recoverable = we can assign each amount to a unique nearest column
    assignments = {}
    for amt, x_pos in amount_positions:
        nearest_role = min(
            money_columns_sorted,
            key=lambda mc: abs(x_pos - mc[1].x_center)
        )[0]
        assignments[amt] = nearest_role

    unique_roles = set(assignments.values())
    is_recoverable = len(unique_roles) == len(amount_positions)
    if is_recoverable:
        recoverable += 1
        print(f"  [OK] RECOVERABLE -> {assignments}")
    else:
        print(f"  [X] CONFLICT -> multiple amounts map to same column: {assignments}")

# ── 5. Recovery estimate ──────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: RECOVERY ESTIMATE")
print("=" * 70)
print(f"Total mismatched rows:       {total_mismatch}")
print(f"Recoverable via position:    {recoverable}")
print(f"Recovery rate:               {recoverable/total_mismatch*100:.0f}%" if total_mismatch else "N/A")
print(f"Current assigned txns:       0")
print(f"Projected assigned txns:     {recoverable}")
