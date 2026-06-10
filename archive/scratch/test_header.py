"""Quick test: run header extraction on all OCR JSON files."""

import json
import logging
from pathlib import Path

from src.schema import OCRToken
from archive.header_extractor import extract_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

outputs = Path("outputs")
for jf in sorted(outputs.glob("*_ocr.json")):
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)

    tokens = [
        OCRToken(text=t["text"], confidence=t["confidence"], bbox=tuple(t["bbox"]))
        for t in data["tokens"]
    ]

    result = extract_metadata(tokens)
    m = result.metadata
    c = result.confidence

    print(f"\n{'='*60}")
    print(f"FILE: {jf.name}")
    print(f"  Bank:       {str(m.bank_name or '-'):<30} (conf={c['bank_name']:.4f})")
    print(f"  Holder:     {str(m.account_holder or '-'):<30} (conf={c['account_holder']:.4f})")
    print(f"  Account:    {str(m.account_number or '-'):<30} (conf={c['account_number']:.4f})")
    print(f"  Period:     {str(m.statement_period_from or '-')} -> {str(m.statement_period_to or '-'):<15} (conf={c['statement_period_from']:.4f})")
    print(f"  Currency:   {str(m.currency or '-'):<30} (conf={c['currency']:.4f})")
