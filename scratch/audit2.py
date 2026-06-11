import json
from pathlib import Path
from collections import defaultdict

from src.layout_detector import detect_layout, load_ocr_tokens
from src.transaction_parser import DATE_RE, MONEY_RE, _is_header_or_footer
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.row_classifier import RowClassifier
from src.transaction_detector import TransactionCandidateDetector
from src.date_validator import validate_date_candidate
from src.semantic_assigner import assign_rows

def main():
    stages = {
        "OCR": {"in": 0, "out": 0, "rejected": 0},
        "Row Reconstruction": {"in": 0, "out": 0, "rejected": 0},
        "Row Classification": {"in": 0, "out": 0, "rejected": 0},
        "Transaction Candidate Detection": {"in": 0, "out": 0, "rejected": 0},
        "Date Extraction": {"in": 0, "out": 0, "rejected": 0},
        "Transaction Parsing": {"in": 0, "out": 0, "rejected": 0},
        "Semantic Assignment": {"in": 0, "out": 0, "rejected": 0},
    }
    
    statements = {}
    classifier = RowClassifier()
    detector = TransactionCandidateDetector()
    
    semantic_rejections = []
    date_failures = defaultdict(int)

    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists(): continue
            
        tokens = load_ocr_tokens(cached_path)
        stages["OCR"]["in"] += len(tokens)
        stages["OCR"]["out"] += len(tokens)
        
        stages["Row Reconstruction"]["in"] += len(tokens)
        raw_rows = reconstruct_rows(tokens)
        stages["Row Reconstruction"]["out"] += len(raw_rows)
        
        layout = detect_layout(tokens, raw_rows)
        
        rows, _ = normalize_date_strings(raw_rows)
        rows, _ = normalize_amount_strings(rows)
        
        parsed_rows_list = []
        assigned_txns_count = 0
        
        for idx, row in enumerate(rows):
            text = row.text
            
            stages["Row Classification"]["in"] += 1
            classification = classifier.classify(text)
            if not classification.is_transaction_candidate:
                stages["Row Classification"]["rejected"] += 1
                continue
            stages["Row Classification"]["out"] += 1
            
            stages["Transaction Candidate Detection"]["in"] += 1
            detection = detector.detect(text)
            if not detection.is_transaction_candidate:
                stages["Transaction Candidate Detection"]["rejected"] += 1
                continue
            stages["Transaction Candidate Detection"]["out"] += 1
            
            stages["Date Extraction"]["in"] += 1
            primary_date_match = DATE_RE.search(text)
            if not primary_date_match:
                stages["Date Extraction"]["rejected"] += 1
                date_failures["no_date_detected"] += 1
                continue
                
            date_str = primary_date_match.group(0)
            is_valid, _ = validate_date_candidate(date_str)
            if not is_valid:
                stages["Date Extraction"]["rejected"] += 1
                date_failures["date_parse_failed"] += 1
                continue
                
            stages["Date Extraction"]["out"] += 1
            
            stages["Transaction Parsing"]["in"] += 1
            amounts = MONEY_RE.findall(text)
            if not amounts:
                stages["Transaction Parsing"]["rejected"] += 1
                continue
            if _is_header_or_footer(text):
                stages["Transaction Parsing"]["rejected"] += 1
                continue
                
            # Valid ParsedRow
            stages["Transaction Parsing"]["out"] += 1
            
            from src.transaction_parser import ParsedRow
            description = text
            parsed_row = ParsedRow(raw_text=text, date=date_str, amounts=amounts, description=description, source_row_index=idx)
            parsed_rows_list.append(parsed_row)
            
        stages["Semantic Assignment"]["in"] += len(parsed_rows_list)
        results = assign_rows(parsed_rows_list, rows, layout)
        
        for ar in results:
            if any(v is not None for v in ar.assigned_values.values()):
                stages["Semantic Assignment"]["out"] += 1
                assigned_txns_count += 1
            else:
                stages["Semantic Assignment"]["rejected"] += 1
                semantic_rejections.append({
                    "text": ar.raw_text,
                    "amounts": ar.raw_amounts,
                    "reason": ar.reason
                })
                
        statements[statement_id] = {
            "reconstructed_rows": len(rows),
            "parsed_rows": len(parsed_rows_list),
            "assigned_transactions": assigned_txns_count
        }

    out = {
        "stages": stages,
        "statements": statements,
        "semantic_rejections": semantic_rejections,
        "date_failures": dict(date_failures)
    }
    
    with open("scratch/audit_results2.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
