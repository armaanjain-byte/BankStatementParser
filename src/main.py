"""
main.py — End-to-end OCR and extraction runner for the Bank Statement Parser.

Runs the complete pipeline:
  Image -> OCR -> Row Reconstruction -> Layout Detection -> 
  Transaction Parsing -> Semantic Assignment -> Header Extraction -> 
  Validation -> BankStatement

Outputs a serialized BankStatement to JSON.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from src.header_extractor import extract_metadata
from src.layout_detector import detect_layout
from src.ocr import extract_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

INPUT_DIR = Path("test_set")
OUTPUT_DIR = Path("statement_outputs")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# ──────────────────────────────────────────────
#  Logging setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pipeline processing
# ──────────────────────────────────────────────

def process_statement(image_path: Path, gpu: bool = True) -> BankStatement:
    """Run the entire end-to-end extraction pipeline on a single image."""
    
    # 1. OCR
    tokens = extract_tokens(image_path, gpu=gpu)
    
    # 2. Row Reconstruction
    rows = reconstruct_rows(tokens)
    
    # 2.5 Normalization
    rows, _ = normalize_date_strings(rows)
    
    # 3. Layout Detection
    layout = detect_layout(tokens, rows)
    
    # 4. Transaction Parsing
    parsed_rows = parse_rows(rows)
    
    # 5. Semantic Assignment
    assignment_report = build_semantic_assignment_report(
        parsed_rows, rows, layout, statement_id=image_path.stem
    )
    
    # Extract successfully assigned transactions
    transactions = []
    for r in assignment_report.rows:
        if any(v is not None for v in r.assigned_values.values()):
            transactions.append(r.transaction)
            
    # 6. Header Extraction
    extraction_result = extract_metadata(tokens)
    metadata = extraction_result.metadata
    
    # 7. Build preliminary BankStatement
    statement = BankStatement(metadata=metadata, transactions=transactions)
    
    # 8. Validation
    validation_report = validate_statement(statement, layout, assignment_report)
    statement.validation = validation_report
    
    return statement


def _output_path_for(image_path: Path) -> Path:
    """Map an image filename to its JSON output path."""
    return OUTPUT_DIR / f"{image_path.stem}_statement.json"


# ──────────────────────────────────────────────
#  Main Batch Runner
# ──────────────────────────────────────────────

def run(*, gpu: bool = True) -> None:
    """Process every image in INPUT_DIR and write BankStatement JSON to OUTPUT_DIR."""

    if not INPUT_DIR.exists():
        logger.error("Input directory not found: %s", INPUT_DIR)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        logger.warning("No images found in %s", INPUT_DIR)
        sys.exit(0)

    logger.info("Found %d image(s) in %s", len(images), INPUT_DIR)
    logger.info("-" * 60)

    batch_start = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        img_start = time.perf_counter()
        
        try:
            statement = process_statement(image_path, gpu=gpu)
            
            out_path = _output_path_for(image_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(statement.to_dict(), f, indent=2, ensure_ascii=False)
                
            elapsed = time.perf_counter() - img_start
            
            logger.info(
                "[%d/%d] %-20s | txns: %4d | passed: %5s | %.2fs",
                idx,
                len(images),
                image_path.name,
                statement.transaction_count,
                statement.validation.passed,
                elapsed,
            )
        except Exception as e:
            logger.exception("Failed to process %s: %s", image_path.name, e)

    batch_elapsed = time.perf_counter() - batch_start

    logger.info("-" * 60)
    logger.info("Done. %d image(s) processed in %.2fs.", len(images), batch_elapsed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch E2E: test_set/ images -> statement_outputs/ JSON.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )
    args = parser.parse_args()

    run(gpu=not args.no_gpu)