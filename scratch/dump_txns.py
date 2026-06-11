import json
from pathlib import Path
from src.evaluation import evaluate_pipeline, FailureReporter
from src.layout_detector import detect_layout, load_ocr_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings

def dump_state(output_file: str):
    state = {}
    total_parsed = 0
    total_assigned = 0
    total_conf = 0.0
    count = 0
    
    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists(): continue
            
        tokens = load_ocr_tokens(cached_path)
        raw_rows = reconstruct_rows(tokens)
        layout = detect_layout(tokens, raw_rows)
        
        rows, _ = normalize_date_strings(raw_rows)
        rows, _ = normalize_amount_strings(rows)
        
        parsed_rows = parse_rows(rows)
        ar = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id=statement_id)
        
        valid_assigned = [r for r in ar.rows if r.transaction and any(v is not None for v in r.assigned_values.values())]
        
        statement = BankStatement(
            transactions=[r.transaction for r in valid_assigned]
        )
        
        val_rep = validate_statement(statement, layout, ar)
        
        total_parsed += len(parsed_rows)
        total_assigned += len(valid_assigned)
        total_conf += val_rep.confidence
        count += 1
        
        txns = []
        for r in valid_assigned:
            txns.append({
                "raw_text": r.raw_text,
                "date": r.transaction.date,
                "amount": r.transaction.amount,
                "debit": r.transaction.debit,
                "credit": r.transaction.credit
            })
            
        state[statement_id] = {
            "parsed_rows": len(parsed_rows),
            "assigned_transactions": len(valid_assigned),
            "validation_confidence": val_rep.confidence,
            "transactions": txns
        }
        
    state["_summary"] = {
        "parsed_rows": total_parsed,
        "assigned_transactions": total_assigned,
        "validation_confidence": total_conf / count if count else 0
    }
        
    with open(output_file, "w") as f:
        json.dump(state, f, indent=2)

if __name__ == "__main__":
    import sys
    dump_state(sys.argv[1])
