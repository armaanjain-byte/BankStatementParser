import json

targets = ['11','13','16','66','90','92','97']
for t in targets:
    data = json.load(open(f'out/statement_outputs/{t}_statement.json'))
    issues = [i['issue_type'] for i in data.get('validation',{}).get('issues',[])]
    meta = data.get('metadata',{})
    bank = meta.get('bank_name')
    print(f'stmt {t}: issues={issues}  bank={repr(bank)}')
