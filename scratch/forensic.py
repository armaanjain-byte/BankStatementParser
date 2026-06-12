import json
from collections import defaultdict

with open('out/evaluation_report.json') as f:
    report = json.load(f)

with open('out/failure_reports.json') as f:
    failures = json.load(f)

# Collect metrics per statement
metrics_by_id = {m['statement_id']: m for m in report['statement_metrics']}

# Group failures by statement
fails_by_stmt = defaultdict(lambda: defaultdict(list))
for f in failures:
    fails_by_stmt[f['statement_id']][f['failure_type']].append(f['candidate'])

print("--- SPECIFIC STATEMENTS ---")
for sid in ['66', '70', '72', '80']:
    m = metrics_by_id.get(sid, {})
    print(f"\nStatement {sid} (Parsed: {m.get('parsed_row_count')}, Assigned: {m.get('assigned_transaction_count')}, Layout Conf: {m.get('layout_detection_confidence')})")
    for ftype, cands in fails_by_stmt[sid].items():
        print(f"  {ftype} ({len(cands)}):")
        for c in cands[:5]:
            print(f"    {repr(c)}")

print("\n--- STATEMENTS WITH PARSED > 0 AND ASSIGNED == 0 ---")
for m in report['statement_metrics']:
    if m['parsed_row_count'] > 0 and m['assigned_transaction_count'] == 0:
        sid = m['statement_id']
        print(f"Statement {sid} (Parsed: {m['parsed_row_count']}, Layout Conf: {m['layout_detection_confidence']})")
        for ftype, cands in fails_by_stmt[sid].items():
            print(f"  {ftype} ({len(cands)}) - Sample: {repr(cands[0]) if cands else ''}")

print("\n--- STATEMENTS WITH HIGH LAYOUT CONF (>0.8) BUT 0 PARSED ---")
for m in report['statement_metrics']:
    if m['layout_detection_confidence'] > 0.8 and m['parsed_row_count'] == 0:
        sid = m['statement_id']
        print(f"Statement {sid} (Layout Conf: {m['layout_detection_confidence']}, Reconstructed: {m['reconstructed_row_count']})")
        for ftype, cands in fails_by_stmt[sid].items():
            print(f"  {ftype} ({len(cands)}) - Sample: {repr(cands[0]) if cands else ''}")

print("\n--- TOP UNKNOWN FAILURES ---")
unk_sorted = sorted(report['statement_metrics'], key=lambda x: x['unknown_failure_count'], reverse=True)
for m in unk_sorted[:5]:
    if m['unknown_failure_count'] > 0:
        sid = m['statement_id']
        print(f"Statement {sid}: {m['unknown_failure_count']} unknown failures")
        for c in fails_by_stmt[sid]['unknown_failure']:
            print(f"  {repr(c)}")

