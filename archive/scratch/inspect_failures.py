import json
from collections import defaultdict

with open("failure_reports.json", "r", encoding="utf-8") as f:
    failures = json.load(f)

by_type = defaultdict(list)
for f in failures:
    by_type[f["failure_type"]].append(f)

for ftype, items in by_type.items():
    print(f"=== FAILURE TYPE: {ftype} ({len(items)} items) ===")
    for item in items:
        print(f"  [{item['statement_id']}] {repr(item['candidate'])}")
