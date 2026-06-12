import json
data = json.load(open('out/evaluation_report.json'))
unks = [m for m in data['statement_metrics'] if m['unknown_failure_count'] > 0 and m['parsed_row_count'] != m['assigned_transaction_count']]
for m in unks:
    print(f"Stmt {m['statement_id']}: Unk={m['unknown_failure_count']}, Parsed={m['parsed_row_count']}, Assigned={m['assigned_transaction_count']}, Layout={m['layout_detection_confidence']}")
