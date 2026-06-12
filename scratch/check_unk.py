import json
data = json.load(open('out/evaluation_report.json'))
for m in sorted(data['statement_metrics'], key=lambda x: x['unknown_failure_count'], reverse=True)[:5]:
    print(f"Stmt {m['statement_id']}: {m['unknown_failure_count']} unknown failures, {m['no_date_detected_count']} no date, {m['insufficient_numeric_values_count']} insufficient")
