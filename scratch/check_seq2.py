import json

fails = json.load(open('out/failure_reports.json'))

stmts = ['72', '70', '5', '80', '25', '71', '20', '23', '61', '83']

for sid in stmts:
    print(f"\n--- Statement {sid} Sequence (including no_date) ---")
    sid_fails = [f for f in fails if f['statement_id'] == sid and f['failure_type'] in ['insufficient_numeric_values', 'date_parse_failed', 'no_date_detected']]
    for f in sid_fails:
        print(f"{f['failure_type']:<30} | {f['candidate']}")
