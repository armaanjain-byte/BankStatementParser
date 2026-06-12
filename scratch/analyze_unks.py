import json
import re
from src.constants import DATE_RE, MONEY_RE
from src.transaction_parser import HEADER_FOOTER_KEYWORDS
from src.date_validator import validate_date_candidate

def is_header_or_footer(text):
    t = text.lower()
    return any(kw in t for kw in HEADER_FOOTER_KEYWORDS)

def analyze_unknowns(sid):
    try:
        data = json.load(open(f'dataset/{sid}.json'))
    except:
        return
        
    print(f"\n--- Statement {sid} ---")
    from src.transaction_reconstructor import reconstruct_rows
    from src.schema import OCRToken
    tokens = [OCRToken(**t) for t in data['tokens']]
    rows = reconstruct_rows(tokens)
    
    from src.transaction_detector import TransactionCandidateDetector
    from src.row_classifier import RowClassifier
    detector = TransactionCandidateDetector()
    classifier = RowClassifier()
    
    unks = 0
    for row in rows:
        text = row.text
        if not classifier.classify(text).is_transaction_candidate: continue
        if not detector.detect(text).is_transaction_candidate: continue
        
        has_date = bool(DATE_RE.search(text))
        has_amount = bool(MONEY_RE.search(text))
        
        if has_date and has_amount:
            # Does it pass parse_rows logic?
            primary_date_match = None
            for candidate in DATE_RE.finditer(text):
                is_valid, _ = validate_date_candidate(candidate.group(0))
                if is_valid:
                    primary_date_match = candidate
                    break
                    
            amounts = MONEY_RE.findall(text)
            
            if primary_date_match and amounts:
                if is_header_or_footer(text):
                    print(f"HEADER/FOOTER SKIP: {text}")
                    unks += 1
                else:
                    # It parsed successfully! If it's an unknown failure, it must be an assignment failure.
                    # But Statement 3, 25, 54 had Parsed == Assigned, so this shouldn't happen!
                    # Wait, Statement 3 Parsed 10, Assigned 10.
                    pass 
            else:
                if not primary_date_match:
                    print(f"VALIDATOR REJECTED DATE: {DATE_RE.search(text).group(0)} in text: {text}")
                    unks += 1
                elif not amounts:
                    pass
                    
    print(f"Total Unknown Failures explained: {unks}")

for sid in ['3', '25', '54', '7', '30']:
    analyze_unknowns(sid)
