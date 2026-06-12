import json
import logging
import sys
sys.path.append('.')
from pathlib import Path
from src.evaluation import FailureReporter, evaluate_pipeline
from src.header_extractor import extract_metadata
from src.layout_detector import detect_layout
from src.ocr import extract_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings

logging.basicConfig(level=logging.ERROR)

def run_pipeline_on_image(image_path: Path):
    statement_id = image_path.stem
    failure_reporter = FailureReporter(Path("temp_failures.json"))
    
    tokens = extract_tokens(image_path, gpu=True, preprocess=False)
    rows = reconstruct_rows(tokens)
    rows, _ = normalize_date_strings(rows)
    rows, _ = normalize_amount_strings(rows)
    layout = detect_layout(tokens, rows)
    parsed_rows = parse_rows(rows)
    assignment_report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id=statement_id)
    transactions = [r.transaction for r in assignment_report.rows if any(v is not None for v in r.assigned_values.values())]

    return {
        'token_count': len(tokens),
        'row_count': len(rows),
        'parsed_count': len(parsed_rows),
        'assigned_count': len(transactions),
        'tokens': [t.text for t in tokens],
        'rows': [r.text for r in rows],
        'parsed_rows': [f"{p.date} {p.amounts} {p.description}" for p in parsed_rows]
    }

def main():
    statements_to_test = ['70.jpg', '20.jpg', '11.jpg', '32.jpg', '27.jpg', '82.jpg']
    
    results_asis = {}
    results_disabled = {}
    
    from src.backends.paddleocr_backend import PaddleOCRBackend
    original_is_low_quality = PaddleOCRBackend._is_low_quality
    
    # Run As-Is
    print("Running As-Is (with heuristic)...")
    for s in statements_to_test:
        path = Path('dataset') / s
        if path.exists():
            results_asis[s] = run_pipeline_on_image(path)
            
    # Run with disabled heuristic
    print("Running with Heuristic Disabled...")
    PaddleOCRBackend._is_low_quality = lambda self, path: False
    for s in statements_to_test:
        path = Path('dataset') / s
        if path.exists():
            results_disabled[s] = run_pipeline_on_image(path)
            
    # Restore
    PaddleOCRBackend._is_low_quality = original_is_low_quality
    
    # Dump results
    output = {
        'as_is': {k: {k2: v2 for k2, v2 in v.items() if k2 not in ('tokens', 'rows', 'parsed_rows')} for k, v in results_asis.items()},
        'disabled': {k: {k2: v2 for k2, v2 in v.items() if k2 not in ('tokens', 'rows', 'parsed_rows')} for k, v in results_disabled.items()},
        '70_samples': {
            'as_is': {
                'tokens': results_asis['70.jpg']['tokens'][:20],
                'rows': results_asis['70.jpg']['rows'][:10],
                'parsed_rows': results_asis['70.jpg']['parsed_rows'][:5]
            },
            'disabled': {
                'tokens': results_disabled['70.jpg']['tokens'][:20],
                'rows': results_disabled['70.jpg']['rows'][:10],
                'parsed_rows': results_disabled['70.jpg']['parsed_rows'][:5]
            }
        }
    }
    
    with open('scratch/experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
