import json
from pathlib import Path
from collections import defaultdict

from src.evaluation import FailureReporter, evaluate_pipeline
from src.header_extractor import extract_metadata
from src.layout_detector import detect_layout, load_ocr_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows, DATE_RE
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.row_classifier import RowClassifier
from src.transaction_detector import TransactionCandidateDetector

def main():
    stages = {
        "OCR": {"in": 0, "out": 0, "rejected": 0},
        "Row Reconstruction": {"in": 0, "out": 0, "rejected": 0},
        "Transaction Candidate Detection": {"in": 0, "out": 0, "rejected": 0},
        "Date Extraction": {"in": 0, "out": 0, "rejected": 0},
        "Transaction Parsing": {"in": 0, "out": 0, "rejected": 0},
        "Semantic Assignment": {"in": 0, "out": 0, "rejected": 0},
        "Validation": {"in": 0, "out": 0, "rejected": 0},
    }
    
    statements = {}
    semantic_rejections = defaultdict(list)
    date_failures = defaultdict(list)
    
    failure_counts = defaultdict(int)
    
    classifier = RowClassifier()
    detector = TransactionCandidateDetector()

    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists():
            continue
            
        tokens = load_ocr_tokens(cached_path)
        stages["OCR"]["in"] += len(tokens)
        stages["OCR"]["out"] += len(tokens)
        
        stages["Row Reconstruction"]["in"] += len(tokens)
        rows = reconstruct_rows(tokens)
        stages["Row Reconstruction"]["out"] += len(rows)
        # Assuming rejected = 0 for reconstruction here, since we don't have exact metrics
        
        # Candidate Detection
        stages["Transaction Candidate Detection"]["in"] += len(rows)
        candidate_rows = []
        for r in rows:
            c = classifier.classify(r.text)
            d = detector.detect(r.text)
            if c.is_transaction_candidate and d.is_transaction_candidate:
                candidate_rows.append(r)
        stages["Transaction Candidate Detection"]["out"] += len(candidate_rows)
        stages["Transaction Candidate Detection"]["rejected"] += len(rows) - len(candidate_rows)
        
        # Normalization changes strings
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        
        # Actual pipeline uses parse_rows on normalized rows
        parsed_rows = parse_rows(rows)
        
        stages["Transaction Parsing"]["in"] += len(candidate_rows)
        stages["Transaction Parsing"]["out"] += len(parsed_rows)
        stages["Transaction Parsing"]["rejected"] += len(candidate_rows) - len(parsed_rows)
        
        # Let's see date failures among candidates that were rejected by parsing
        # Actually parse_rows only processes rows that pass classifier and detector.
        # So candidates = rows that passed filters.
        # Why did they fail to become parsed_rows?
        parsed_texts = [pr.raw_text for pr in parsed_rows]
        for r in candidate_rows:
            if r.text not in parsed_texts:
                if DATE_RE.search(r.text):
                    date_failures["date_parse_failed"].append(r.text)
                    failure_counts["date_parse_failed"] += 1
                else:
                    date_failures["no_date_detected"].append(r.text)
                    failure_counts["no_date_detected"] += 1
        
        layout = detect_layout(tokens, rows)
        assignment_report = build_semantic_assignment_report(
            parsed_rows, rows, layout, statement_id=statement_id
        )
        
        stages["Semantic Assignment"]["in"] += len(parsed_rows)
        
        assigned_txns = []
        for ar in assignment_report.rows:
            if ar.transaction and any(v is not None for v in ar.assigned_values.values()):
                assigned_txns.append(ar)
            else:
                # Rejected by semantic assignment
                reason = ar.reason or "Unknown"
                from src.semantic_assigner import _layout_money_columns
                semantic_rejections[reason].append({
                    "text": ar.raw_text,
                    "amounts": ar.raw_amounts,
                    "columns": [(role, c.x_center) for role, c in _layout_money_columns(layout)],
                    "reason": reason
                })
                failure_counts["Semantic Assignment Loss"] += 1
                
        stages["Semantic Assignment"]["out"] += len(assigned_txns)
        stages["Semantic Assignment"]["rejected"] += len(parsed_rows) - len(assigned_txns)
        
        stages["Validation"]["in"] += len(assigned_txns)
        # All assigned txns pass into validation, we just assume none rejected strictly for now
        stages["Validation"]["out"] += len(assigned_txns)
        
        statements[statement_id] = {
            "reconstructed_rows": len(rows),
            "parsed_rows": len(parsed_rows),
            "assigned_transactions": len(assigned_txns)
        }

    out = {
        "stages": stages,
        "statements": statements,
        "semantic_rejections": dict(semantic_rejections),
        "date_failures": dict(date_failures),
        "failure_counts": dict(failure_counts)
    }
    
    with open("scratch/audit_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
