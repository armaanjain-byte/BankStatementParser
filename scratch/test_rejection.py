import json
from pathlib import Path
from collections import Counter
from src.constants import DATE_RE
from src.date_validator import validate_date_candidate
from src.layout_detector import load_ocr_tokens
from src.transaction_reconstructor import reconstruct_rows

def main():
    rejected_rows = []
    rejection_reasons = Counter()
    alphabetic_count = 0
    
    for i in range(23, 29):
        statement_id = str(i)
        cached_path = Path(f"outputs/{statement_id}_ocr.json")
        if not cached_path.exists():
            print(f"Could not find {cached_path}")
            continue
            
        tokens = load_ocr_tokens(cached_path)
        rows = reconstruct_rows(tokens)
        
        for row in rows:
            primary_date_match = DATE_RE.search(row.text)
            if primary_date_match:
                date_str = primary_date_match.group(0)
                is_valid, reason = validate_date_candidate(date_str)
                if not is_valid:
                    rejected_rows.append((row.text, date_str, reason))
                    rejection_reasons[(date_str, reason)] += 1
                    
                    # Check if it's an alphabetic date
                    import re
                    if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", date_str, re.IGNORECASE):
                        alphabetic_count += 1
                        
    for row_text, date_str, reason in rejected_rows:
        print("ROW:")
        print(row_text)
        print("\nEXTRACTED DATE:")
        print(date_str)
        print("\nVALID:")
        print("False")
        print("\nREASON:")
        print(reason)
        print("\n====================================================")
        
    print("\nSummary Table:")
    print(f"{'date_string':<15} | {'rejection_reason':<65} | {'count'}")
    for (date_str, reason), count in rejection_reasons.most_common():
        print(f"{date_str:<15} | {reason:<65} | {count}")
        
    total_rejected = len(rejected_rows)
    print("\n====================================================")
    if total_rejected > 0:
        print(f"Total rejected: {total_rejected}")
        print(f"Alphabetic dates: {alphabetic_count} ({(alphabetic_count/total_rejected)*100:.2f}%)")
        
if __name__ == "__main__":
    main()
