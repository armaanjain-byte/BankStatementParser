"""Analyze Y-coordinate distributions across all OCR JSON files."""
import json
from pathlib import Path

for jf in sorted(Path("outputs").glob("*_ocr.json")):
    data = json.load(open(jf))
    tokens = data["tokens"]
    ys = sorted(set((t["bbox"][1] + t["bbox"][3]) // 2 for t in tokens))
    diffs = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
    small = [d for d in diffs if d <= 10]
    print(f"{jf.name:20s} | y_centers: {len(ys):3d} | "
          f"gaps: min={min(diffs):2d} max={max(diffs):3d} med={sorted(diffs)[len(diffs)//2]:2d} | "
          f"same-row (<=10): {len(small)}/{len(diffs)}")
