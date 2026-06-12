import json
fails = json.load(open('out/failure_reports.json'))
targets = ['11', '13', '16', '66', '90', '92', '97']
for f in fails:
    sid = str(f.get('statement_id'))
    if sid in targets and f.get('failure_type') == 'unknown_failure':
        print(f"stmt {sid} candidate: {repr(f.get('candidate'))}")
