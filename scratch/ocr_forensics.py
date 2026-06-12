import json
import os
import sys

# We will run this script using paddle_env python
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR not installed in this environment.")
    sys.exit(1)

sys.path.append('.')
from src.schema import OCRToken
from src.transaction_reconstructor import Row as ReconstructedRow
from src.transaction_reconstructor import reconstruct_rows
from src.transaction_parser import parse_rows

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def analyze_statement(sid):
    print(f"\n{'='*50}")
    print(f"ANALYZING STATEMENT {sid}")
    print(f"{'='*50}")
    
    img_path = f"dataset/{sid}.jpg"
    result = ocr.ocr(img_path, cls=True)
    
    tokens = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        tokens.append(OCRToken(
            text=text,
            confidence=confidence,
            bbox=(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        ))
        
    print(f"1. OCR Tokens Generated: {len(tokens)}")
    
    # Let's find specific dates or amounts that were lost
    if sid == '72':
        targets = ['2021-03-26', '2021-03-29', '2021-03-30', '0.39', '73.45', '79.10']
        print("\n--- Token Search ---")
        for t in tokens:
            if any(target in t.text for target in targets):
                print(f"Token: '{t.text}' | Box: y0={t.bbox[1]:.1f}, y1={t.bbox[3]:.1f}, x0={t.bbox[0]:.1f}, x1={t.bbox[2]:.1f}")

    # Reconstruct rows
    rows = reconstruct_rows(tokens)
    print(f"\n2. Reconstructed Rows: {len(rows)}")
    
    if sid == '72':
        print("\n--- Relevant Reconstructed Rows ---")
        for i, r in enumerate(rows):
            if any(target in r.text for target in targets):
                print(f"Row {i}: '{r.text}'")

    if sid == '80':
        print("\n--- Reconstructed Rows Sample ---")
        for i, r in enumerate(rows[10:20]):
            print(f"Row {10+i}: '{r.text}'")

for sid in ['72', '80', '70', '97']:
    analyze_statement(sid)
