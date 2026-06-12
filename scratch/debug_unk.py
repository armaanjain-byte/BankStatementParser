import json
from src.evaluation import evaluate_pipeline
from src.constants import DATE_RE, MONEY_RE

# We will just mimic the logic of evaluate_pipeline to find what failed parsing
sids = ['66', '25', '3', '54', '7']

for sid in sids:
    print(f"\n{'='*40}\nStatement {sid}\n{'='*40}")
    try:
        data = json.load(open(f'dataset/{sid}.json'))
        from src.transaction_reconstructor import reconstruct_rows
        from src.schema import OCRToken
        tokens = [OCRToken(**t) for t in data['tokens']]
        rows = reconstruct_rows(tokens)
        
        # normalize
        from src.normalizers.date_normalizer import normalize_date_strings
        from src.normalizers.amount_normalizer import normalize_amount_strings
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        
        from src.transaction_parser import parse_rows
        parsed_rows = parse_rows(rows)
        parsed_indices = {pr.source_row_index for pr in parsed_rows}
        
        from src.layout_detector import BankStatementLayoutDetector
        detector = BankStatementLayoutDetector()
        layout = detector.detect_layout(parsed_rows)
        
        from src.semantic_assigner import assign_semantics
        report = assign_semantics(parsed_rows, layout)
        assigned_indices = {ar.source_row_index for ar in report.rows}
        
        for idx, row in enumerate(rows):
            if idx in assigned_indices:
                continue
            if idx in parsed_indices:
                assigned_row = next((ar for ar in report.rows if ar.source_row_index == idx), None)
                if assigned_row:
                    reason = assigned_row.reason
                    if "amount count does not match" not in reason and reason != "one or more parsed amounts could not be converted":
                        print(f"[UNKNOWN - PARSED BUT ASSIGN FAILED] Reason: {reason} | Text: {row.text}")
                else:
                    print(f"[UNKNOWN - PARSED BUT NOT IN REPORT] Text: {row.text}")
            else:
                has_date = bool(DATE_RE.search(row.text))
                has_amount = bool(MONEY_RE.search(row.text))
                if has_date and has_amount:
                    print(f"[UNKNOWN - HAS DATE & AMOUNT BUT PARSE FAILED] Text: {row.text}")
                elif not has_date:
                    pass # known failure
                elif has_date and not has_amount:
                    pass # known failure
    except Exception as e:
        print(f"Error reading {sid}: {e}")
