import logging
from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows, DATE_RE, MONEY_RE
from src.semantic_assigner import build_semantic_assignment_report

logging.disable(logging.CRITICAL)

def dump_statement_26():
    print("=== STATEMENT 26: SEMANTIC MISMATCH EVIDENCE ===")
    tokens = extract_tokens("test_set/26.jpg")
    rows = reconstruct_rows(tokens)
    layout = detect_layout(tokens, rows)
    parsed = parse_rows(rows)
    report = build_semantic_assignment_report(parsed, rows, layout, statement_id="26")

    header_text = str(layout.header_rows[0]) if layout.header_rows else "None"
    print(f"Header Row: {header_text}")
    print("-" * 50)

    mismatch_count = 0
    for ar in report.rows:
        if not ar.transaction or not any(v is not None for v in ar.assigned_values.values()):
            print(f"\n[Row {ar.source_row_index}]")
            print(f"Text: {ar.raw_text}")
            print(f"Extracted Amounts: {ar.raw_amounts}")
            print(f"Assignment Reason: {ar.reason}")
            mismatch_count += 1

    print(f"\nTotal Unassigned Rows in 26: {mismatch_count}")
    print("================================================\n")


def dump_statement_25():
    print("=== STATEMENT 25: INSUFFICIENT NUMERIC VALUES EVIDENCE ===")
    tokens = extract_tokens("test_set/25.jpg")
    rows = reconstruct_rows(tokens)
    layout = detect_layout(tokens, rows)
    
    header_idx = max((hr.source_row_index for hr in layout.header_rows), default=-1)
    
    count = 0
    for i, row in enumerate(rows):
        if i <= header_idx:
            continue
            
        has_date = bool(DATE_RE.search(row.text))
        has_amount = bool(MONEY_RE.search(row.text))
        
        if has_date and not has_amount:
            print(f"\n--- Incident #{count+1} (Row {i}) ---")
            start = max(0, i - 2)
            end = min(len(rows), i + 3)
            for j in range(start, end):
                prefix = ">>" if j == i else "  "
                print(f"{prefix} Row {j:02d}: {rows[j].text}")
                
            count += 1
            if count >= 20:
                break
                
    print(f"\nTotal Insufficient Numeric Rows Shown: {count}")
    print("=========================================================\n")


if __name__ == "__main__":
    dump_statement_26()
    dump_statement_25()
