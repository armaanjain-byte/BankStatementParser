import json
from pathlib import Path

from src.layout_detector import load_ocr_tokens, detect_layout
from src.transaction_reconstructor import reconstruct_rows
from src.normalizers.amount_normalizer import normalize_amount_strings
from src.normalizers.date_normalizer import normalize_date_strings
from src.transaction_parser import parse_rows
from src.semantic_assigner import assign_rows

sid = "90"
ocr_cache = Path(f"out/statement_outputs/{sid}_ocr.json")
# Wait, I don't have OCR cache for 90. But I can synthesize the layout detection from row texts!
# Wait, I don't have row texts for statement 90. But I do have failure candidates.
# Let's just create a synthetic statement 90 row:
from src.schema import OCRToken
from tests.fixtures.ocr_tokens import make_token

tokens = [
    make_token("2026-11-20", x=10, y=100, w=80, h=20),
    make_token("DEBIT", x=120, y=100, w=40, h=20),
    make_token("1,200.00", x=300, y=100, w=60, h=20),
    make_token("3,400.00", x=400, y=100, w=60, h=20)
]

rows = reconstruct_rows(tokens)
print(f"Reconstructed: {rows[0].text}")
rows, _ = normalize_date_strings(rows)
rows, _ = normalize_amount_strings(rows)
layout = detect_layout(tokens, rows)

print(f"Layout detection method: {layout.detection_method}")
from src.semantic_assigner import _layout_money_columns
print(f"Money columns detected: {_layout_money_columns(layout)}")

parsed = parse_rows(rows)
results = assign_rows(parsed, rows, layout)
print(f"Assigned values: {results[0].assigned_values}")
