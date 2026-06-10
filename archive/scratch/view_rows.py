import json
from pathlib import Path
from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout

input_folder = Path("test_set")
images = sorted(p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"})

for image_path in images:
    sid = image_path.stem
    print(f"\n=================== STATEMENT {sid} ===================")
    tokens = extract_tokens(image_path, gpu=False)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    
    header_idx = -1
    if layout.header_rows:
        header_idx = max(hr.source_row_index for hr in layout.header_rows)
        
    print(f"Header Index: {header_idx}")
    money_cols = []
    if layout.debit_column: money_cols.append(("debit", layout.debit_column.x_center))
    if layout.credit_column: money_cols.append(("credit", layout.credit_column.x_center))
    if layout.balance_column: money_cols.append(("balance", layout.balance_column.x_center))
    if layout.amount_column: money_cols.append(("amount", layout.amount_column.x_center))
    print(f"Layout Money Columns: {money_cols}")
    
    for idx, r in enumerate(rows):
        prefix = "  [HDR]" if idx <= header_idx else f"  [{idx:2d}]"
        print(f"{prefix} text: {repr(r.text)} | orig: {repr(r.original_text)}")
