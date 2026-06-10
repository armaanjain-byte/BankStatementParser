from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows
from src.semantic_assigner import build_semantic_assignment_report

image_path = "test_set/24.jpg"
tokens = extract_tokens(image_path, gpu=False, preprocess=True, min_confidence=0.0)
rows = reconstruct_rows(tokens)
rows, _ = normalize_date_strings(rows)
rows, _ = normalize_amount_strings(rows)
layout = detect_layout(tokens, rows)
parsed_rows = parse_rows(rows)

print("=== LAYOUT ===")
print("Header Row Index:", [hr.source_row_index for hr in layout.header_rows])
print("Layout Details:", layout.to_dict())

print("\n=== RECONSTRUCTED ROWS ===")
for idx, r in enumerate(rows):
    print(f"  [{idx:2d}] Text: {repr(r.text)}")

print("\n=== PARSED ROWS ===")
for pr in parsed_rows:
    print(f"  Row {pr.source_row_index}: Date={repr(pr.date)} | Amounts={pr.amounts} | Desc={repr(pr.description)}")

report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id="24")
print("\n=== ASSIGNMENT REPORT ===")
print(f"Assigned row count: {report.assigned_row_count}")
for r in report.rows:
    print(f"  Row {r.source_row_index}: Assigned={r.assigned_values} | Reason={repr(r.reason)} | Tx={r.transaction}")
