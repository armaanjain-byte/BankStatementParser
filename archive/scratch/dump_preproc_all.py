from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows
from src.semantic_assigner import build_semantic_assignment_report
from pathlib import Path

images = sorted(Path("test_set").glob("*.jpg"))

print("=== DETAILED PREPROCESSING FAILURES ===")
for img in images:
    sid = img.stem
    print(f"\nStatement {sid}:")
    tokens = extract_tokens(img, gpu=False, preprocess=True, min_confidence=0.0)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    parsed_rows = parse_rows(rows)
    
    print(f"  Tokens: {len(tokens)} | Rows: {len(rows)} | Parsed Rows: {len(parsed_rows)}")
    print("  Sample Rows:")
    for idx, r in enumerate(rows):
        # Print a few rows from transaction zone
        if 10 <= idx < 15:
            print(f"    [{idx:2d}] {repr(r.text)}")
