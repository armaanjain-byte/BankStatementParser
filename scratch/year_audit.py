import json
from datetime import datetime
from collections import Counter
from pathlib import Path
from src.layout_detector import load_ocr_tokens, detect_layout
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.transaction_parser import parse_rows
from src.semantic_assigner import assign_rows

def audit_years():
    results = {}
    current_year = datetime.now().year
    
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
        assigned = assign_rows(parsed_rows, rows, layout)
        
        valid_assigned = [ar for ar in assigned if any(v is not None for v in ar.assigned_values.values())]
        
        # Gather all valid years to find dominant year
        years = []
        out_of_range = []
        
        for ar in valid_assigned:
            date_str = ar.transaction.date
            if date_str and len(date_str) >= 4:
                y = int(date_str.split("-")[0])
                years.append(y)
                
                if y < 1990 or y > current_year + 2:
                    out_of_range.append({
                        "raw_text": ar.raw_text,
                        "original_ocr_text": rows[ar.source_row_index].original_text if hasattr(rows[ar.source_row_index], 'original_text') else "",
                        "parsed_date": str(date_str),
                        "assigned_amount": ar.transaction.amount,
                        "year": y
                    })
                    
        year_counts = dict(Counter(years))
        
        if out_of_range:
            results[statement_id] = {
                "dominant_years": year_counts,
                "out_of_range_transactions": out_of_range
            }
            
    with open("scratch/year_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    audit_years()
