import json
from pathlib import Path

from src.evaluation import FailureReporter, evaluate_pipeline
from src.header_extractor import extract_metadata
from src.layout_detector import detect_layout, load_ocr_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings

def main():
    print("Post-Fix Metrics:")
    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists():
            print(f"Statement {statement_id} cache missing.")
            continue
            
        tokens = load_ocr_tokens(cached_path)
        rows = reconstruct_rows(tokens)
        
        rows, normalized_date_count = normalize_date_strings(rows)
        rows, normalized_amount_count = normalize_amount_strings(rows)
        
        layout = detect_layout(tokens, rows)
        parsed_rows = parse_rows(rows)
        assignment_report = build_semantic_assignment_report(
            parsed_rows, rows, layout, statement_id=statement_id
        )
        
        transactions = [
            r.transaction for r in assignment_report.rows
            if any(v is not None for v in r.assigned_values.values())
        ]
        
        print(f"Statement {statement_id}:")
        print(f"  reconstructed_rows: {len(rows)}")
        print(f"  parsed_rows: {len(parsed_rows)}")
        print(f"  assigned_transactions: {len(transactions)}")
        print()

if __name__ == "__main__":
    main()
