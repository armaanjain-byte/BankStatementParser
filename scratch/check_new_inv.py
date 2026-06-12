import json
import collections

fails = json.load(open('out/failure_reports.json'))

stmts = collections.defaultdict(list)
for f in fails:
    if f['failure_type'] == 'insufficient_numeric_values':
        stmts[f['statement_id']].append(f)

for sid, failures in sorted(stmts.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\nStatement {sid}: {len(failures)} INV failures")
    for f in failures[:5]:
        print(f"    INV: {f['candidate']}")
