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


@dataclass
class PipelineMetrics:
    """Metrics for a single statement pipeline execution."""
    statement_id: str
    ocr_token_count: int
    reconstructed_row_count: int
    parsed_row_count: int
    assigned_transaction_count: int
    header_extraction_success: bool
    layout_detection_confidence: float
    validation_confidence: float
    date_parse_failure_count: int
    amount_parse_failure_count: int
    semantic_assignment_failure_count: int
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

    def report_failure(self, failure_type: str, raw_text: str, statement_id: str) -> None:
        """Report an anonymized failure.
        
        Args:
            failure_type: E.g., 'date_parse', 'amount_parse', 'semantic_assignment'.
            raw_text: The specific token or constrained substring that failed.
                      NEVER pass full descriptions or entire row texts here to avoid PII.
            statement_id: The ID of the statement.
        """
        # Ensure we are not storing massive strings that might contain PII
        if len(raw_text) > 50:
            raw_text = raw_text[:47] + "..."
            
        self.failures.append({
            "failure_type": failure_type,
            "raw_text": raw_text,
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
        # parsed_row has .source_row_index which corresponds to the original Row
        # Wait, ParsedRow doesn't store a reference to the Row object, just the index.
        # So we can look up the original row.
        row_idx = parsed_row.source_row_index
        if 0 <= row_idx < len(rows):
            orig_row = rows[row_idx]
            has_date_now = bool(DATE_RE.search(orig_row.text))
            had_date_before = bool(DATE_RE.search(orig_row.original_text))
            if has_date_now and not had_date_before:
                recovered_date_parse_count += 1
    
    # Calculate date parse failures
    # We consider a date parse failure if a row has a monetary amount (looks like a transaction)
    # but the regex failed to find a date, occurring after the header row.
    date_parse_failures = 0
    header_idx = -1
    if layout.header_rows:
        header_idx = max(hr.source_row_index for hr in layout.header_rows)
        
    for idx, row in enumerate(rows):
        if idx <= header_idx:
            continue
        has_amount = bool(MONEY_RE.search(row.text))
        has_date = bool(DATE_RE.search(row.text))
        if has_amount and not has_date:
            date_parse_failures += 1
            # To anonymize, we grab only the first few tokens that might be the date
            # instead of the full row text which contains descriptions.
            first_few_tokens = " ".join(t.text for t in row.tokens[:3])
            failure_reporter.report_failure(
                failure_type="date_parse",
                raw_text=first_few_tokens,
                statement_id=statement_id
            )

    # Calculate amount parse failures & semantic assignment failures
    amount_parse_failures = 0
    semantic_assignment_failures = 0
    
    for row_res in assignment_report.rows:
        if row_res.reason == "one or more parsed amounts could not be converted":
            amount_parse_failures += 1
            # To anonymize, only log the raw amounts that failed
            failure_reporter.report_failure(
                failure_type="amount_parse",
                raw_text=str(row_res.raw_amounts),
                statement_id=statement_id
            )
        elif "amount count does not match" in row_res.reason:
            semantic_assignment_failures += 1
            failure_reporter.report_failure(
                failure_type="semantic_assignment",
                raw_text=f"amounts: {len(row_res.raw_amounts)}",
                statement_id=statement_id
            )

    metrics = PipelineMetrics(
        statement_id=statement_id,
        ocr_token_count=len(tokens),
        reconstructed_row_count=len(rows),
        parsed_row_count=len(parsed_rows),
        assigned_transaction_count=assignment_report.assigned_row_count,
        header_extraction_success=bool(statement.metadata.account_number),
        layout_detection_confidence=layout.confidence_scores.get("table_header", 0.0),
        validation_confidence=validation_report.confidence,
        date_parse_failure_count=date_parse_failures,
        amount_parse_failure_count=amount_parse_failures,
        semantic_assignment_failure_count=semantic_assignment_failures,
        normalized_date_count=normalized_date_count,
        recovered_date_parse_count=recovered_date_parse_count,
    )
    
    logger.info("Evaluated pipeline for %s: %d assigned txns, %.2f confidence", 
                statement_id, metrics.assigned_transaction_count, metrics.validation_confidence)
    
    return metrics
