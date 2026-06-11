import json
from pathlib import Path
from src.layout_detector import load_ocr_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.row_classifier import RowClassifier
from src.transaction_detector import TransactionCandidateDetector
from src.transaction_parser import parse_rows
from src.layout_detector import detect_layout
from src.semantic_assigner import assign_rows
from src.validator import validate_statement
from src.schema import BankStatement

def run_pipeline():
    results = {}
    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists():
            continue
            
        tokens = load_ocr_tokens(cached_path)
        raw_rows = reconstruct_rows(tokens)
        layout = detect_layout(tokens, raw_rows)
        
        rows, _ = normalize_date_strings(raw_rows)
        rows, _ = normalize_amount_strings(rows)
        
        parsed_rows = parse_rows(rows)
        assigned = assign_rows(parsed_rows, rows, layout)
        
        valid_assigned = [ar for ar in assigned if any(v is not None for v in ar.assigned_values.values())]
        
        statement = BankStatement(transactions=[ar.transaction for ar in valid_assigned])
        validated_stmt = validate_statement(statement)
        
        txns = []
        for v_txn in validated_stmt.transactions:
            # Match back to AssignmentResult to get raw_text
            raw_text = ""
            for ar in valid_assigned:
                if ar.transaction.date == v_txn.date and ar.transaction.description == v_txn.description:
                    raw_text = ar.raw_text
                    break
            txns.append({
                "date": v_txn.date,
                "description": v_txn.description,
                "amount": v_txn.amount,
                "debit": v_txn.debit,
                "credit": v_txn.credit,
                "balance": v_txn.balance,
                "validation_confidence": v_txn.validation_confidence,
                "validation_issues": v_txn.validation_issues,
                "raw_text": raw_text
            })
            
        results[statement_id] = {
            "parsed_rows": len(parsed_rows),
            "assigned_transactions": len(valid_assigned),
            "validation_confidence": validated_stmt.validation_confidence,
            "transactions": txns
        }
    return results

if __name__ == "__main__":
    out = run_pipeline()
    with open("scratch/current_state.json", "w") as f:
        json.dump(out, f, indent=2)
