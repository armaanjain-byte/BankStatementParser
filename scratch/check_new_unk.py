import json
data = json.load(open('out/evaluation_report.json'))
for m in sorted(data['statement_metrics'], key=lambda x: x['unknown_failure_count'], reverse=True)[:10]:
    if m['unknown_failure_count'] > 0:
        print(f"Stmt {m['statement_id']}: {m['unknown_failure_count']} unknown failures, Layout Conf: {m['layout_detection_confidence']}, Parsed: {m['parsed_row_count']}, Assigned: {m['assigned_transaction_count']}")
