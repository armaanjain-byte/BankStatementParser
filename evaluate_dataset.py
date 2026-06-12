"""
evaluate_dataset.py - High-level runner for dataset-wide observability.

This script processes a folder of statement images through the entire
pipeline, generates individual BankStatement JSONs, aggregates telemetry
into an evaluation_report.json, and writes an anonymized failure_reports.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from src.evaluation import FailureReporter, PipelineMetrics, evaluate_pipeline
from src.header_extractor import extract_metadata
from src.layout_detector import detect_layout
from src.ocr import extract_tokens
from src.schema import BankStatement
from src.semantic_assigner import build_semantic_assignment_report
from src.transaction_parser import parse_rows
from src.transaction_reconstructor import reconstruct_rows
from src.validator import validate_statement
from src.normalizer import normalize_date_strings, normalize_amount_strings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}


def _evaluate_one_image(
    image_path: Path,
    statement_outputs_dir: Path,
    no_gpu: bool,
) -> tuple[str, PipelineMetrics | None, list[dict], str | None]:
    statement_id = image_path.stem
    failure_reporter = FailureReporter(Path("__worker_failures_unused.json"))

    try:
        if image_path.suffix.lower() == ".pdf":
            from src.pdf_converter import pdf_to_images

            tokens = []
            rows = []
            for page_image in pdf_to_images(image_path):
                page_tokens = extract_tokens(page_image, gpu=not no_gpu, preprocess=False)
                tokens.extend(page_tokens)
                rows.extend(reconstruct_rows(page_tokens))
        else:
            cached_path = Path(f"outputs/{statement_id}_ocr.json")
            if cached_path.exists():
                from src.layout_detector import load_ocr_tokens
                tokens = load_ocr_tokens(cached_path)
            else:
                tokens = extract_tokens(image_path, gpu=not no_gpu)
            rows = reconstruct_rows(tokens)

        rows, normalized_date_count = normalize_date_strings(rows)
        rows, normalized_amount_count = normalize_amount_strings(rows)

        layout = detect_layout(tokens, rows)
        parsed_rows = parse_rows(rows)
        assignment_report = build_semantic_assignment_report(
            parsed_rows,
            rows,
            layout,
            statement_id=statement_id,
        )

        transactions = [
            r.transaction for r in assignment_report.rows
            if any(v is not None for v in r.assigned_values.values())
        ]

        metadata = extract_metadata(tokens).metadata
        statement = BankStatement(metadata=metadata, transactions=transactions)
        validation_report = validate_statement(statement, layout, assignment_report)
        statement.validation = validation_report

        out_json = statement_outputs_dir / f"{statement_id}_statement.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(statement.to_dict(), f, indent=2, ensure_ascii=False)

        metrics = evaluate_pipeline(
            statement_id=statement_id,
            tokens=tokens,
            rows=rows,
            parsed_rows=parsed_rows,
            layout=layout,
            assignment_report=assignment_report,
            statement=statement,
            validation_report=validation_report,
            failure_reporter=failure_reporter,
            normalized_date_count=normalized_date_count,
            normalized_amount_count=normalized_amount_count,
        )
        return statement_id, metrics, failure_reporter.failures, None
    except Exception as exc:
        return statement_id, None, [], str(exc)


def evaluate_dataset(
    input_folder: Path,
    output_folder: Path,
    no_gpu: bool = False,
    workers: int = 2,
    clear_cache: bool = False,
) -> None:
    """Run the pipeline on all images, collecting telemetry and outputs.

    Use --workers 1 to disable parallelism, which is useful for debugging.
    """
    
    if not input_folder.exists() or not input_folder.is_dir():
        logger.error("Input folder does not exist: %s", input_folder)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    statement_outputs_dir = output_folder / "statement_outputs"
    statement_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if clear_cache:
        logger.info("Clearing cached OCR results...")
        for p in Path("outputs").glob("*_ocr.json"):
            try:
                p.unlink()
            except OSError:
                pass
    
    failure_reporter = FailureReporter(output_folder / "failure_reports.json")
    
    images = sorted(
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        logger.warning("No images found in %s", input_folder)
        sys.exit(0)

    logger.info("Found %d image(s) for evaluation in %s", len(images), input_folder)
    
    metrics_list: List[PipelineMetrics] = []
    
    start_time = time.perf_counter()
    
    max_workers = max(1, workers)
    if max_workers == 1:
        for idx, image_path in enumerate(images, start=1):
            logger.info("[%d/%d] Processing %s...", idx, len(images), image_path.name)
            statement_id, metrics, failures, error = _evaluate_one_image(
                image_path,
                statement_outputs_dir,
                no_gpu,
            )
            if metrics:
                metrics_list.append(metrics)
            for failure in failures:
                failure_reporter.failures.append(failure)
            if error:
                logger.error("Failed to evaluate %s: %s", image_path.name, error)
                failure_reporter.report_failure("unhandled_exception", error, statement_id)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    _evaluate_one_image,
                    image_path,
                    statement_outputs_dir,
                    no_gpu,
                ): image_path
                for image_path in images
            }
            for idx, future in enumerate(as_completed(future_to_image), start=1):
                image_path = future_to_image[future]
                statement_id, metrics, failures, error = future.result()
                if metrics:
                    metrics_list.append(metrics)
                for failure in failures:
                    failure_reporter.failures.append(failure)
                if error:
                    logger.error("Failed to evaluate %s: %s", image_path.name, error)
                    failure_reporter.report_failure("unhandled_exception", error, statement_id)
                logger.info("[%d/%d] Finished %s", idx, len(images), image_path.name)

    # Save failures
    failure_reporter.save()
    
    elapsed = time.perf_counter() - start_time
    
    # Aggregate Metrics
    aggregate_report(metrics_list, output_folder, elapsed)
    
    logger.info("Evaluation complete! Results saved to %s", output_folder)


def aggregate_report(metrics_list: List[PipelineMetrics], output_folder: Path, elapsed_time: float) -> None:
    """Compute aggregate statistics and write the evaluation report."""
    
    if not metrics_list:
        return
        
    total_statements = len(metrics_list)
    total_txns = sum(m.assigned_transaction_count for m in metrics_list)
    
    header_successes = sum(1 for m in metrics_list if m.header_extraction_success)
    
    validation_confidences = [m.validation_confidence for m in metrics_list]
    layout_confidences = [m.layout_detection_confidence for m in metrics_list]
    
    report = {
        "summary": {
            "total_statements_processed": total_statements,
            "total_transactions_assigned": total_txns,
            "total_processing_time_seconds": round(elapsed_time, 2),
            "header_extraction_success_rate": round(header_successes / total_statements, 4),
            "average_validation_confidence": round(statistics.mean(validation_confidences), 4),
            "average_layout_confidence": round(statistics.mean(layout_confidences), 4),
        },
        "normalization": {
            "total_normalized_date_count": sum(m.normalized_date_count for m in metrics_list),
            "total_recovered_date_parse_count": sum(m.recovered_date_parse_count for m in metrics_list),
        },
        "amount_normalization": {
            "total_normalized_amount_count": sum(m.normalized_amount_count for m in metrics_list),
            "total_recovered_amount_parse_count": sum(m.recovered_amount_parse_count for m in metrics_list),
        },
        "statement_metrics": [m.to_dict() for m in metrics_list]
    }
    
    evaluation_report_path = output_folder / "evaluation_report.json"
    with open(evaluation_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    # Generate pipeline_loss_report.json
    failure_counts = {
        "no_date_detected": sum(m.no_date_detected_count for m in metrics_list),
        "date_parse_failed": sum(m.date_parse_failed_count for m in metrics_list),
        "insufficient_numeric_values": sum(m.insufficient_numeric_values_count for m in metrics_list),
        "amount_extraction_failed": sum(m.amount_extraction_failed_count for m in metrics_list),
        "semantic_mismatch": sum(m.semantic_mismatch_count for m in metrics_list),
        "unknown": sum(m.unknown_failure_count for m in metrics_list)
    }
    
    # Sort descending
    ranked_failures = dict(sorted(failure_counts.items(), key=lambda item: item[1], reverse=True))
    
    loss_report = {
        "ranked_failure_categories": ranked_failures,
        "statement_conversions": []
    }
    
    for m in metrics_list:
        loss_report["statement_conversions"].append({
            "statement_id": m.statement_id,
            "reconstructed_rows": m.reconstructed_row_count,
            "parsed_rows": m.parsed_row_count,
            "assigned_transactions": m.assigned_transaction_count,
            "parse_conversion_rate": m.parse_conversion_rate,
            "assignment_conversion_rate": m.assignment_conversion_rate
        })
        
    loss_report_path = output_folder / "pipeline_loss_report.json"
    with open(loss_report_path, "w", encoding="utf-8") as f:
        json.dump(loss_report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Bank Statement Parser dataset.")
    parser.add_argument("input_folder", type=Path, help="Folder containing statement images.")
    parser.add_argument("--output_folder", type=Path, default=Path("out"), help="Where to output reports.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for OCR.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker processes. Use 1 to disable parallelism.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached OCR results in the outputs directory.")
    
    args = parser.parse_args()
    evaluate_dataset(args.input_folder, args.output_folder, args.no_gpu, args.workers, args.clear_cache)
