import sys
from pathlib import Path
from src.main import process_statement
from src.ocr_paddle import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows
from src.semantic_assigner import build_semantic_assignment_report

def debug_statement(stmt_id: str):
    image_path = Path(f"test_set/{stmt_id}.jpg")
    if not image_path.exists():
        # Maybe it's a PDF or png?
        for ext in (".png", ".pdf", ".jpeg"):
            p = Path(f"test_set/{stmt_id}{ext}")
            if p.exists():
                image_path = p
                break
    
    print(f"=== Debugging Statement {stmt_id} ({image_path.name}) ===")
    tokens = extract_tokens(image_path, gpu=False, preprocess=False, min_confidence=0.10)
    print(f"Extracted {len(tokens)} tokens.")
    rows = reconstruct_rows(tokens)
    print(f"Reconstructed {len(rows)} rows.")
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    print("Layout Columns:")
    for role in ("date", "description", "debit", "credit", "balance", "amount"):
        col = getattr(layout, f"{role}_column")
        if col:
            print(f"  {role}: center={col.x_center}, text={col.source_text}")
        else:
            print(f"  {role}: None")
            
    parsed_rows = parse_rows(rows)
    print(f"Parsed {len(parsed_rows)} transaction rows.")
    report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id=stmt_id)
    
    for idx, r in enumerate(report.rows):
        print(f"Row {idx+1}: text='{r.raw_text}'")
        print(f"  amounts={r.raw_amounts} -> parsed={r.assigned_values}")
        print(f"  reason: {r.reason}")

if __name__ == "__main__":
    debug_statement("25")
    debug_statement("28")
