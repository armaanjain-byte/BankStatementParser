"""Verify row reconstruction output for file 28."""

import json
import re

# ── Load data ─────────────────────────────────
with open("outputs/28_ocr.json", encoding="utf-8") as f:
    ocr_data = json.load(f)

with open("outputs/28_rows.json", encoding="utf-8") as f:
    rows_data = json.load(f)

rows = rows_data["rows"]
total_ocr_tokens = ocr_data["token_count"]
total_rows = rows_data["row_count"]

# ── Section 2: Summary stats ──────────────────
token_counts = [r["token_count"] for r in rows]
avg_tokens = sum(token_counts) / len(token_counts)

print("=" * 70)
print("SECTION 2: Row Reconstruction Summary (file 28)")
print("=" * 70)
print(f"  Total OCR tokens:      {total_ocr_tokens}")
print(f"  Total reconstructed rows: {total_rows}")
print(f"  Average tokens per row:   {avg_tokens:.2f}")
print(f"  Minimum tokens in a row:  {min(token_counts)}")
print(f"  Maximum tokens in a row:  {max(token_counts)}")

# ── Section 3 & 4: Rows 10-20 ────────────────
print()
print("=" * 70)
print("SECTION 3 & 4: Rows 10 through 20")
print("=" * 70)
for r in rows:
    idx = r["row_number"]
    if 10 <= idx <= 20:
        print(f"  Row {idx:2d} | tokens: {r['token_count']:2d} | {r['text']}")

# ── Section 5: Pattern detection ──────────────
DATE_RE = re.compile(
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{2,4})?\b",
    re.IGNORECASE,
)

MONEY_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*\.\d{2}\b"   # 1,234.56 or 226.14
)

rows_with_date = []
rows_with_money = []
rows_with_both = []

for r in rows:
    has_date = bool(DATE_RE.search(r["text"]))
    has_money = bool(MONEY_RE.search(r["text"]))
    if has_date:
        rows_with_date.append(r["row_number"])
    if has_money:
        rows_with_money.append(r["row_number"])
    if has_date and has_money:
        rows_with_both.append(r["row_number"])

print()
print("=" * 70)
print("SECTION 5: Pattern Counts")
print("=" * 70)
print(f"  Rows containing a date:           {len(rows_with_date)}  {rows_with_date}")
print(f"  Rows containing monetary values:  {len(rows_with_money)}  {rows_with_money}")
print(f"  Rows containing BOTH:             {len(rows_with_both)}  {rows_with_both}")

# ── Section 6: Transaction candidates ─────────
print()
print("=" * 70)
print("SECTION 6: Transaction Candidate Rows (date + money)")
print("=" * 70)
for r in rows:
    if r["row_number"] in rows_with_both:
        print(f"  Row {r['row_number']:2d} | tokens: {r['token_count']:2d} | conf: {r['avg_confidence']:.4f} | {r['text']}")

print()
print(f"Total transaction candidates: {len(rows_with_both)} / {total_rows} rows")
