import json
from pathlib import Path

# Statements with parsed_rows > 0 but assigned_transactions = 0
TARGETS = ["11", "13", "16", "66", "90", "92", "97"]

print("="*80)
print("ASSIGNMENT FAILURE TRACE")
print("="*80)

for sid in TARGETS:
    path = Path(f'out/statement_outputs/{sid}_statement.json')
    if not path.exists():
        print(f"Skipping {sid} - file not found")
        continue
        
    data = json.load(open(path))
    print(f"\nSTATEMENT {sid}")
    
    # 1. Validation issues
    issues = [i['issue_type'] for i in data.get('validation', {}).get('issues', [])]
    print(f"  Validation Issues: {issues}")
    
    # 2. Assignment Report (if available, the full report is in semantic_assignment_report.json,
    # but let's check what's in the statement output first)
    
    # Actually, we need to look at the semantic assignment reports!
    # Let's see if we have them. The output JSON has transactions = [] 
    
    report_path = Path(f'out/statement_outputs/{sid}_semantic_assignment_report.json')
    if not report_path.exists():
        # Maybe they are not saved there? 
        print(f"  No semantic assignment report found at {report_path}")
        continue
        
    report = json.load(open(report_path))
    
    layout = report.get('layout_summary', {})
    print(f"  Detection Method: {layout.get('detection_method')}")
    print(f"  Layout Confidence: {layout.get('confidence_scores')}")
    
    cols = layout.get('columns', {})
    money_cols = {k: v for k, v in cols.items() if k in ['amount', 'debit', 'credit', 'balance'] and v is not None}
    print(f"  Money Columns Detected: {list(money_cols.keys())}")
    
    rows = report.get('rows', [])
    print(f"  Parsed Rows: {len(rows)}")
    
    if len(rows) > 0:
        # Just show the first row's reason
        print(f"  Sample rejection reason: {rows[0].get('reason')}")

