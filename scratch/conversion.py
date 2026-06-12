import json
data = json.load(open('out/pipeline_loss_report.json'))
conversions = data['statement_conversions']
# Find the statements with the worst assignment_conversion_rate
# and statements with the worst parse_conversion_rate where reconstructed_rows > 10
worst_assign = sorted([s for s in conversions if s['reconstructed_rows'] > 10], key=lambda x: x['assignment_conversion_rate'])
print('WORST ASSIGNMENT RATES:')
for s in worst_assign[:5]:
    print(f"Stmt {s['statement_id']}: Parsed={s['parsed_rows']}, Assigned={s['assigned_transactions']}")
    
worst_parse = sorted([s for s in conversions if s['reconstructed_rows'] > 10], key=lambda x: x['parse_conversion_rate'])
print('\nWORST PARSE RATES:')
for s in worst_parse[:5]:
    print(f"Stmt {s['statement_id']}: Recon={s['reconstructed_rows']}, Parsed={s['parsed_rows']}")
