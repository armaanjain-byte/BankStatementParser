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
import argparse
from pathlib import Path

from archive.header_extractor import extract_metadata
from src.layout_detector import detect_layout
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings

# ──────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

INPUT_DIR = Path("test_set")
OUTPUT_DIR = Path("statement_outputs")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}

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

def _process_page(
    image_input,
    *,
    statement_id: str,
    gpu: bool = True,
    preprocess: bool = False,
    use_paddle: bool = True,
) -> BankStatement:
    """Run the end-to-end extraction pipeline on a single image/page."""
    
    # 1. OCR
    if use_paddle:
        from src.ocr_paddle import extract_tokens
    else:
        from src.ocr import extract_tokens
        
    tokens = extract_tokens(image_input, gpu=gpu, preprocess=preprocess)
    
    # 2. Row Reconstruction
    rows = reconstruct_rows(tokens)
    
    # 2.5 Normalization
    rows, normalized_date_count = normalize_date_strings(rows)
    rows, normalized_amount_count = normalize_amount_strings(rows)
    logger.debug(
        "Normalized rows: dates=%d, amounts=%d",
        normalized_date_count,
        normalized_amount_count,
    )
    
    # 3. Layout Detection
    layout = detect_layout(tokens, rows)
    
    # 4. Transaction Parsing
    parsed_rows = parse_rows(rows)
    
    # 5. Semantic Assignment
    assignment_report = build_semantic_assignment_report(
        parsed_rows, rows, layout, statement_id=statement_id
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


def process_statement(
    image_path: Path,
    gpu: bool = True,
    preprocess: bool = False,
    use_paddle: bool = True,
) -> BankStatement:
    """Run the entire end-to-end extraction pipeline on a single image or PDF."""
    image_path = Path(image_path)
    if image_path.suffix.lower() != ".pdf":
        return _process_page(
            image_path,
            statement_id=image_path.stem,
            gpu=gpu,
            preprocess=preprocess,
            use_paddle=use_paddle,
        )

    from src.pdf_converter import pdf_to_images

    pages = pdf_to_images(image_path)
    page_statements = [
        _process_page(
            page,
            statement_id=f"{image_path.stem}_page_{idx}",
            gpu=gpu,
            preprocess=False,
            use_paddle=use_paddle,
        )
        for idx, page in enumerate(pages, start=1)
    ]

    if not page_statements:
        return BankStatement()

    combined = BankStatement(
        metadata=page_statements[0].metadata,
        transactions=[
            transaction
            for statement in page_statements
            for transaction in statement.transactions
        ],
        validation=page_statements[0].validation,
    )
    return combined


def _output_path_for(image_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Map an image filename to its JSON output path."""
    return output_dir / f"{image_path.stem}_statement.json"


# ──────────────────────────────────────────────
#  Main Batch Runner
# ──────────────────────────────────────────────

def run(
    *,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gpu: bool = True,
    preprocess: bool = False,
    use_paddle: bool = True,
) -> None:
    """Process every image in input_dir and write BankStatement JSON to output_dir."""

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        logger.warning("No images found in %s", input_dir)
        sys.exit(0)

    logger.info("Found %d image(s) in %s", len(images), input_dir)
    logger.info("-" * 60)

    batch_start = time.perf_counter()

    for idx, image_path in enumerate(images, start=1):
        img_start = time.perf_counter()
        
        try:
            statement = process_statement(
                image_path,
                gpu=gpu,
                preprocess=preprocess,
                use_paddle=use_paddle,
            )
            
            out_path = _output_path_for(image_path, output_dir)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bank statement OCR extraction.",
    )
    parser.add_argument("--input", type=Path, default=INPUT_DIR, help="Input folder.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable OpenCV preprocessing.",
    )
    parser.add_argument(
        "--no-paddle",
        action="store_true",
        help="Disable PaddleOCR and fallback to EasyOCR.",
    )
    parser.add_argument(
        "--single",
        type=Path,
        default=None,
        help="Process one image/PDF and print JSON to stdout.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.single:
        statement = process_statement(
            args.single,
            gpu=not args.no_gpu,
            preprocess=not args.no_preprocess,
            use_paddle=not args.no_paddle,
        )
        print(json.dumps(statement.to_dict(), indent=2, ensure_ascii=False))
    else:
        run(
            input_dir=args.input,
            output_dir=args.output,
            gpu=not args.no_gpu,
            preprocess=not args.no_preprocess,
            use_paddle=not args.no_paddle,
        )
