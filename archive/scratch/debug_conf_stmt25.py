from archive.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows
from src.semantic_assigner import build_semantic_assignment_report

def run_for_conf(min_conf):
    tokens = extract_tokens("test_set/25.jpg", gpu=False, preprocess=False, min_confidence=min_conf)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    parsed_rows = parse_rows(rows)
    report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id="25")
    
    assigned_rows = []
    for r in report.rows:
        if r.transaction and any(v is not None for v in r.assigned_values.values()):
            assigned_rows.append((r.source_row_index, r.transaction.date, r.transaction.amount or r.transaction.debit or r.transaction.credit))
    return assigned_rows

assigned_0 = run_for_conf(0.0)
assigned_30 = run_for_conf(0.30)

set_0 = { (date, amt) for idx, date, amt in assigned_0 }
set_30 = { (date, amt) for idx, date, amt in assigned_30 }

lost = set_0 - set_30

print(f"=== CONFIDENCE COMPARE FOR STATEMENT 25 ===")
print(f"Assigned with 0.0 conf: {len(assigned_0)}")
for idx, date, amt in assigned_0:
    print(f"  Row {idx}: Date={date} | Amt={amt}")
    
print(f"\nAssigned with 0.30 conf: {len(assigned_30)}")
for idx, date, amt in assigned_30:
    print(f"  Row {idx}: Date={date} | Amt={amt}")

print(f"\nLost transactions ({len(lost)}):")
for date, amt in lost:
    # Find the corresponding row in assigned_0
    row_info = [r for r in assigned_0 if r[1] == date and r[2] == amt][0]
    print(f"  Row {row_info[0]}: Date={date} | Amt={amt}")
    
    # Print the tokens of this row to see their confidence scores
    tokens = extract_tokens("test_set/25.jpg", gpu=False, preprocess=False, min_confidence=0.0)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    failed_row = rows[row_info[0]]
    print(f"    Raw Text: {repr(failed_row.text)}")
    print(f"    Tokens confidence scores:")
    for t in failed_row.tokens:
        print(f"      Token: {repr(t.text)} | Conf: {t.confidence:.4f}")
