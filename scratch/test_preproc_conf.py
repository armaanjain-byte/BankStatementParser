import json
from pathlib import Path
from src.ocr import extract_tokens
from src.transaction_reconstructor import reconstruct_rows
from src.normalizer import normalize_date_strings, normalize_amount_strings
from src.layout_detector import detect_layout
from src.transaction_parser import parse_rows
from src.semantic_assigner import build_semantic_assignment_report

input_folder = Path("test_set")
images = sorted(p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"})

def run_pipeline(image_path, preprocess, min_confidence):
    try:
        # Extract tokens with specified params
        tokens = extract_tokens(image_path, gpu=False, preprocess=preprocess, min_confidence=min_confidence)
        token_count = len(tokens)
        
        # Run reconstruction
        rows = reconstruct_rows(tokens)
        
        # Normalization
        rows, _ = normalize_date_strings(rows)
        rows, _ = normalize_amount_strings(rows)
        
        # Layout & Parse
        layout = detect_layout(tokens, rows)
        parsed_rows = parse_rows(rows)
        
        # Assigner
        assignment_report = build_semantic_assignment_report(parsed_rows, rows, layout, statement_id=image_path.stem)
        assigned_count = assignment_report.assigned_row_count
        
        return token_count, assigned_count
    except Exception as e:
        print(f"Error for {image_path.name}: {e}")
        return 0, 0

print("=== FORENSIC CONFIGURATION BENCHMARK ===")
print(f"{'Stmt':<6} | {'NoPre+NoConf':<15} | {'NoPre+Conf':<15} | {'Pre+NoConf':<15} | {'Pre+Conf':<15}")
print("-" * 75)

for img in images:
    sid = img.stem
    # 1. No Preprocess, No Confidence (min_confidence=0.0)
    tok1, ass1 = run_pipeline(img, preprocess=False, min_confidence=0.0)
    
    # 2. No Preprocess, Min Confidence = 0.30
    tok2, ass2 = run_pipeline(img, preprocess=False, min_confidence=0.30)
    
    # 3. Preprocess, No Confidence (min_confidence=0.0)
    tok3, ass3 = run_pipeline(img, preprocess=True, min_confidence=0.0)
    
    # 4. Preprocess, Min Confidence = 0.30
    tok4, ass4 = run_pipeline(img, preprocess=True, min_confidence=0.30)
    
    print(f"{sid:<6} | {f'{tok1} t ({ass1} tx)':<15} | {f'{tok2} t ({ass2} tx)':<15} | {f'{tok3} t ({ass3} tx)':<15} | {f'{tok4} t ({ass4} tx)':<15}")
