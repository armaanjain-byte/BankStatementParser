import json
import collections

with open('out/evaluation_report.json') as f:
    report = json.load(f)

with open('out/failure_reports.json') as f:
    failures = json.load(f)

# Rank statements by potential recovered transactions (unknown, no_date_detected, date_parse_failed)
stmt_metrics = {}
for metric in report['statement_metrics']:
    sid = metric['statement_id']
    lost = (metric['no_date_detected_count'] + 
            metric['unknown_failure_count'] + 
            metric['date_parse_failed_count'] + 
            metric['insufficient_numeric_values_count'])
    stmt_metrics[sid] = {
        'lost': lost,
        'assigned': metric['assigned_transaction_count'],
        'parsed': metric['parsed_row_count'],
        'unknown': metric['unknown_failure_count'],
        'no_date': metric['no_date_detected_count'],
        'date_fail': metric['date_parse_failed_count'],
        'insufficient': metric['insufficient_numeric_values_count']
    }

top_lost = sorted(stmt_metrics.items(), key=lambda x: x[1]['lost'], reverse=True)[:15]

print(f"{'Stmt':<6} | {'Lost':<5} | {'Assign':<6} | {'Parsed':<6} | {'Unk':<4} | {'NoDt':<4} | {'DtFl':<4} | {'Insf':<4}")
print('-' * 65)
for sid, metrics in top_lost:
    print(f"{sid:<6} | {metrics['lost']:>5} | {metrics['assigned']:>6} | {metrics['parsed']:>6} | {metrics['unknown']:>4} | {metrics['no_date']:>4} | {metrics['date_fail']:>4} | {metrics['insufficient']:>4}")
