import json
from src.transaction_parser import parse_rows
from src.evaluation import evaluate_pipeline

class MockRow:
    def __init__(self, text):
        self.text = text

rows = [
    MockRow("2021-03-26 PURCHASE"),
    MockRow("0.39 13")
]

parsed = parse_rows(rows)
print(f"Parsed rows count: {len(parsed)}")
for p in parsed:
    print(f"Index: {p.source_row_index}, Text: {p.raw_text}, Date: {p.date}")
