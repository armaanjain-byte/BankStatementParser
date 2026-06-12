import json
from collections import defaultdict

with open('out/failure_reports.json') as f:
    failures = json.load(f)

# Group by statement ID and failure type
stmt_failures = defaultdict(lambda: defaultdict(list))
for f in failures:
    sid = str(f.get('statement_id'))
    ftype = f.get('failure_type')
    cand = f.get('candidate')
    stmt_failures[sid][ftype].append(cand)

top_sids = ['3', '66', '72', '70', '10', '25', '42', '80']

for sid in top_sids:
    print(f"\n{'='*80}\nSTATEMENT {sid}\n{'='*80}")
    for ftype, cands in stmt_failures[sid].items():
        print(f"\n--- {ftype.upper()} ({len(cands)}) ---")
        for c in cands[:10]:  # Show up to 10 examples
            print(f"  {repr(c)}")
        if len(cands) > 10:
            print(f"  ... and {len(cands)-10} more")

