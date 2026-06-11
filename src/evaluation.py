"""
evaluation.py - Framework for pipeline evaluation and anonymized failure reporting.

Collects metrics across the entire extraction pipeline and provides mechanisms
to report failures without exposing Personally Identifiable Information (PII).

Bug fixes in this version
─────────────────────────
1. FailureReporter accumulation bug (Critical):
   The previous version loaded existing failure_reports.json on __init__ and
   appended to it. Every evaluation run doubled the file contents. After two
   runs the forensic data was polluted: failure counts were doubled, and records
   from old code versions (with different schemas) contaminated current analysis.
   Fix: FailureReporter now starts with an empty list unconditionally. The file
   is overwritten on save(), not appended to.

2. Failure record key standardised to 'candidate' (Medium):
   The previous version wrote {"candidate": ...} in evaluation.py but old records
   in the accumulated file used {"raw_text": ...}. forensic_analyzer.py papered
   over this with failure.get("candidate", failure.get("raw_text", "")).
   Fix: All records now use "candidate" consistently. forensic_analyzer.py fallback
   is retained for backward compatibility with any existing archived reports.

3. evaluation.py imports DATE_RE from transaction_parser (important):
   DATE_RE is now expanded in transaction_parser.py. evaluation.py imports it
   directly, so DATE_CANDIDATE_RE and recovered_date_parse_count logic
   automatically benefits from the expanded pattern without any change here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from src.constants import DATE_RE, MONEY_RE
from src.layout_detector import LayoutDefinition
from src.schema import BankStatement, OCRToken, ValidationReport
from src.semantic_assigner import SemanticAssignmentReport
from src.transaction_parser import ParsedRow
from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)


DATE_CANDIDATE_RE = re.compile(
    r"\b\d{1,2}[\s/\-\.]+(?:\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s/\-\.]+(?:\d{2,4})\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
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
    
    # Amount normalization metrics
    normalized_amount_count: int
    recovered_amount_parse_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class FailureReporter:
    """Collects and writes anonymized failure reports.
    
    Each call to evaluate_dataset produces a clean report. Previous runs
    are NOT loaded or merged — the output file is always overwritten.
    This ensures forensic data is never polluted by records from prior
    code versions or duplicate runs.
    """
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        # Always start clean. Do NOT load prior runs.
        # Previous version loaded existing file here, causing accumulation across runs.
        self.failures: List[dict] = []

    def report_failure(self, failure_type: str, candidate: str, statement_id: str) -> None:
        """Report an anonymized failure.
        
        Args:
            failure_type: Category of failure (e.g. 'date_parse_failed', 'no_date_detected').
            candidate:    The specific token or short substring that failed.
                          NEVER pass full row text here — truncate to avoid PII.
                          Automatically truncated to 50 chars.
            statement_id: The statement image stem (e.g. '24', '27').
        """
        # Truncate to avoid storing PII in bulk text that happened to contain a date
        if len(candidate) > 50:
            candidate = candidate[:47] + "..."
            
        self.failures.append({
            "failure_type": failure_type,
            "candidate": candidate,        # Standardised key — was "raw_text" in old code
            "statement_id": statement_id,
        })
        
    def save(self) -> None:
        """Write current run's failures to disk, overwriting any previous file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.failures, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d failure records -> %s", len(self.failures), self.output_path)


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
    normalized_amount_count: int = 0,
) -> PipelineMetrics:
    """Compute metrics for a single statement and log failures."""
    
    # Calculate recovered date parse counts:
    # A row is "recovered by normalizer" if DATE_RE matches its current text
    # but would NOT have matched its original text (before ordinal stripping).
    recovered_date_parse_count = 0
    recovered_amount_parse_count = 0
    for parsed_row in parsed_rows:
        row_idx = parsed_row.source_row_index
        if 0 <= row_idx < len(rows):
            orig_row = rows[row_idx]
            has_date_now = bool(DATE_RE.search(orig_row.text))
            had_date_before = bool(DATE_RE.search(orig_row.original_text))
            if has_date_now and not had_date_before:
                recovered_date_parse_count += 1
            # Check if amount normalization recovered an amount
            has_amount_now = bool(MONEY_RE.search(orig_row.text))
            had_amount_before = bool(MONEY_RE.search(orig_row.original_text))
            if has_amount_now and not had_amount_before:
                recovered_amount_parse_count += 1
                
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
    assigned_indices = {
        r.source_row_index
        for r in assignment_report.rows
        if r.transaction and any(v is not None for v in r.assigned_values.values())
    }
    
    # Analyze all rows in the transaction zone
    from src.transaction_detector import TransactionCandidateDetector
    detector = TransactionCandidateDetector()
    
    for idx, row in enumerate(rows):
        if idx <= header_idx:
            continue
            
        if idx in assigned_indices:
            continue  # Row succeeded — skip
            
        if not detector.detect(row.text).is_transaction_candidate:
            continue  # Skip true non-transaction rows from failure reporting
            
        # Row failed to become an assigned transaction. Classify why.
        parsed_row_obj = next(
            (pr for pr in parsed_rows if pr.source_row_index == idx), None
        )
        
        if not parsed_row_obj:
            # Failed at parse stage — never became a ParsedRow
            has_date = bool(DATE_RE.search(row.text))
            has_amount = bool(MONEY_RE.search(row.text))
            
            if has_date and not has_amount:
                # Date present, no amount — row is likely split across lines
                insufficient_numeric_values_count += 1
                failure_reporter.report_failure(
                    "insufficient_numeric_values",
                    DATE_RE.search(row.text).group(0),
                    statement_id,
                )
            elif not has_date:
                # No date at all — check if there's a date-like candidate
                candidate_match = DATE_CANDIDATE_RE.search(row.text)
                if candidate_match:
                    # Has something date-like that DATE_RE still rejected = OCR corruption
                    date_parse_failed_count += 1
                    failure_reporter.report_failure(
                        "date_parse_failed",
                        candidate_match.group(0),
                        statement_id,
                    )
                else:
                    # No date-like pattern at all = non-transaction row or severe corruption
                    no_date_detected_count += 1
                    first_few = " ".join(t.text for t in row.tokens[:3])
                    failure_reporter.report_failure(
                        "no_date_detected",
                        first_few,
                        statement_id,
                    )
            else:
                unknown_failure_count += 1

        else:
            # Became a ParsedRow but failed semantic assignment
            assigned_row = next(
                (ar for ar in assignment_report.rows if ar.source_row_index == idx),
                None,
            )
            
            if assigned_row:
                reason = assigned_row.reason
                if reason == "one or more parsed amounts could not be converted":
                    amount_extraction_failed_count += 1
                    failure_reporter.report_failure(
                        "amount_extraction_failed",
                        str(assigned_row.raw_amounts),
                        statement_id,
                    )
                elif "amount count does not match" in reason:
                    semantic_mismatch_count += 1
                    failure_reporter.report_failure(
                        "semantic_mismatch",
                        f"amounts: {len(assigned_row.raw_amounts)}",
                        statement_id,
                    )
                else:
                    unknown_failure_count += 1
            else:
                unknown_failure_count += 1

    # Conversion Rates
    reconstructed_rows = len(rows)
    parsed_rows_count = len(parsed_rows)
    assigned_count = assignment_report.assigned_row_count
    
    parse_conversion_rate = (
        parsed_rows_count / reconstructed_rows if reconstructed_rows > 0 else 0.0
    )
    assignment_conversion_rate = (
        assigned_count / parsed_rows_count if parsed_rows_count > 0 else 0.0
    )

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
        normalized_amount_count=normalized_amount_count,
        recovered_amount_parse_count=recovered_amount_parse_count,
    )
    
    logger.info(
        "Evaluated pipeline for %s: %d assigned txns, %.2f confidence",
        statement_id,
        metrics.assigned_transaction_count,
        metrics.validation_confidence,
    )
    
    return metrics