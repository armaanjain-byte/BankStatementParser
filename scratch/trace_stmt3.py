import re
from src.constants import DATE_RE
from src.normalizers.date_normalizer import normalize_date_strings
from src.transaction_reconstructor import reconstruct_rows
from src.schema import OCRToken
from tests.fixtures.ocr_tokens import make_token

text = "S9703109702/04/2017 349802030032344:Int.Pd:01 INR"

print("--- DATE_RE BEHAVIOR ---")
match = DATE_RE.search(text)
if match:
    print(f"Matched: {match.group(0)}")
else:
    print("NO MATCH")

# Create a mock row
tokens = [make_token(text, 10, 100, 200, 20)]
rows = reconstruct_rows(tokens)
print(f"\n--- NORMALIZATION BEHAVIOR ---")
normalized_rows, count = normalize_date_strings(rows)
print(f"Count normalized: {count}")
print(f"Resulting text: {normalized_rows[0].text}")
