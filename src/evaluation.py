"""
evaluation.py - Framework for pipeline evaluation and anonymized failure reporting.

Collects metrics across the entire extraction pipeline and provides mechanisms
to report failures without exposing Personally Identifiable Information (PII).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from src.layout_detector import LayoutDefinition
from src.schema import BankStatement, OCRToken, ValidationReport
from src.semantic_assigner import SemanticAssignmentReport
from src.transaction_parser import ParsedRow, MONEY_RE, DATE_RE
from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)


DATE_CANDIDATE_RE = re.compile(
    r"\b\d{1,2}[\s/\-\.]+(?:\d{1,2}|[A-Za-z]{3})[\s/\-\.]+(?:\d{2,4})\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{2,4}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\b",
    re.IGNORECASE
)

@dataclass
class PipelineMetrics:
    """Metrics for a single statement pipeline execution."""
    statement_id: str
    ocr_token_count: int
    reconstructed_row_count: int
    parsed_row_count: int
    assigned_transaction_count: int
    parse_conversion_rate: float
    assignment_conversion_rate: float
    header_extraction_success: bool
    layout_detection_confidence: float
    validation_confidence: float
    
    # Detailed Failure Tracking
    no_date_detected_count: int
    date_parse_failed_count: int
    insufficient_numeric_values_count: int
    amount_extraction_failed_count: int
    semantic_mismatch_count: int
    unknown_failure_count: int
    
    normalized_date_count: int
    recovered_date_parse_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class FailureReporter:
    """Collects and writes anonymized failure reports."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.failures: List[dict] = []
        
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    self.failures = json.load(f)
            except json.JSONDecodeError:
                self.failures = []

    def report_failure(self, failure_type: str, candidate: str, statement_id: str) -> None:
        """Report an anonymized failure.
        
        Args:
            failure_type: E.g., 'date_parse', 'amount_parse', 'semantic_assignment'.
            candidate: The specific token or constrained substring that failed.
                      NEVER pass full descriptions or entire row texts here to avoid PII.
            statement_id: The ID of the statement.
        """
        # Ensure we are not storing massive strings that might contain PII
        if len(candidate) > 50:
            candidate = candidate[:47] + "..."
            
        self.failures.append({
            "failure_type": failure_type,
            "candidate": candidate,
            "statement_id": statement_id
        })
        
    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.failures, f, indent=2, ensure_ascii=False)


def evaluate_pipeline(
    statement_id: str,
    tokens: List[OCRToken],
    rows: List[Row],
    parsed_rows: List[ParsedRow],
    layout: LayoutDefinition,
    assignment_report: SemanticAssignmentReport,
    statement: BankStatement,
    validation_report: ValidationReport,
    failure_reporter: FailureReporter,
    normalized_date_count: int = 0,
) -> PipelineMetrics:
    """Compute metrics for a single statement and log failures."""
    
    # Calculate recovered date parse counts
    recovered_date_parse_count = 0
    for parsed_row in parsed_rows:
        row_idx = parsed_row.source_row_index
        if 0 <= row_idx < len(rows):
            orig_row = rows[row_idx]
            has_date_now = bool(DATE_RE.search(orig_row.text))
            had_date_before = bool(DATE_RE.search(orig_row.original_text))
            if has_date_now and not had_date_before:
                recovered_date_parse_count += 1
                
    # Define tracking counters
    no_date_detected_count = 0
    date_parse_failed_count = 0
    insufficient_numeric_values_count = 0
    amount_extraction_failed_count = 0
    semantic_mismatch_count = 0
    unknown_failure_count = 0
    
    header_idx = -1
    if layout.header_rows:
        header_idx = max(hr.source_row_index for hr in layout.header_rows)
        
    # Get all row indices that were successfully assigned
    assigned_indices = {r.source_row_index for r in assignment_report.rows if r.transaction and any(v is not None for v in r.assigned_values.values())}
    
    # Analyze all rows in the transaction zone
    for idx, row in enumerate(rows):
        if idx <= header_idx:
            continue
            
        if idx in assigned_indices:
            continue  # Row succeeded!
            
        # If we reach here, the row failed to become an assigned transaction.
        # Find out why:
        
        # Did it even become a ParsedRow?
        parsed_row_obj = next((pr for pr in parsed_rows if pr.source_row_index == idx), None)
        
        if not parsed_row_obj:
            # Failed to parse. Why?
            has_date = bool(DATE_RE.search(row.text))
            has_amount = bool(MONEY_RE.search(row.text))
            
            if has_date and not has_amount:
                insufficient_numeric_values_count += 1
                failure_reporter.report_failure("insufficient_numeric_values", DATE_RE.search(row.text).group(0), statement_id)
            elif not has_date:
                # Did it have a candidate?
                candidate_match = DATE_CANDIDATE_RE.search(row.text)
                if candidate_match:
                    date_parse_failed_count += 1
                    failure_reporter.report_failure("date_parse_failed", candidate_match.group(0), statement_id)
                else:
                    no_date_detected_count += 1
                    # Can't log candidate if none detected, just use first few tokens
                    first_few = " ".join(t.text for t in row.tokens[:3])
                    failure_reporter.report_failure("no_date_detected", first_few, statement_id)
            else:
                unknown_failure_count += 1
        else:
            # It became a ParsedRow, but failed semantic assignment
            assigned_row = next((ar for ar in assignment_report.rows if ar.source_row_index == idx), None)
            
            if assigned_row:
                reason = assigned_row.reason
                if reason == "one or more parsed amounts could not be converted":
                    amount_extraction_failed_count += 1
                    failure_reporter.report_failure("amount_extraction_failed", str(assigned_row.raw_amounts), statement_id)
                elif "amount count does not match" in reason:
                    semantic_mismatch_count += 1
                    failure_reporter.report_failure("semantic_mismatch", f"amounts: {len(assigned_row.raw_amounts)}", statement_id)
                else:
                    unknown_failure_count += 1
            else:
                unknown_failure_count += 1

    # Conversion Rates
    reconstructed_rows = len(rows)
    parsed_rows_count = len(parsed_rows)
    assigned_count = assignment_report.assigned_row_count
    
    parse_conversion_rate = parsed_rows_count / reconstructed_rows if reconstructed_rows > 0 else 0.0
    assignment_conversion_rate = assigned_count / parsed_rows_count if parsed_rows_count > 0 else 0.0

    metrics = PipelineMetrics(
        statement_id=statement_id,
        ocr_token_count=len(tokens),
        reconstructed_row_count=reconstructed_rows,
        parsed_row_count=parsed_rows_count,
        assigned_transaction_count=assigned_count,
        parse_conversion_rate=round(parse_conversion_rate, 2),
        assignment_conversion_rate=round(assignment_conversion_rate, 2),
        header_extraction_success=bool(statement.metadata.account_number),
        layout_detection_confidence=layout.confidence_scores.get("table_header", 0.0),
        validation_confidence=validation_report.confidence,
        no_date_detected_count=no_date_detected_count,
        date_parse_failed_count=date_parse_failed_count,
        insufficient_numeric_values_count=insufficient_numeric_values_count,
        amount_extraction_failed_count=amount_extraction_failed_count,
        semantic_mismatch_count=semantic_mismatch_count,
        unknown_failure_count=unknown_failure_count,
        normalized_date_count=normalized_date_count,
        recovered_date_parse_count=recovered_date_parse_count,
    )
    
    logger.info("Evaluated pipeline for %s: %d assigned txns, %.2f confidence", 
                statement_id, metrics.assigned_transaction_count, metrics.validation_confidence)
    
    return metrics
