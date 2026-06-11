"""
validator.py - Validation layer for analyzing extraction results.

This module evaluates the outputs of the extraction pipeline and reports
quality indicators, generating a ValidationReport with confidence scores
and specific issues.
"""

from __future__ import annotations

import logging
from typing import List

from src.layout_detector import LayoutDefinition
from src.schema import BankStatement, ValidationIssue, ValidationReport
from src.semantic_assigner import SemanticAssignmentReport

logger = logging.getLogger(__name__)


def validate_statement(
    statement: BankStatement,
    layout: LayoutDefinition,
    assignment_report: SemanticAssignmentReport,
) -> ValidationReport:
    """Analyze extraction results and compute confidence and issues."""
    issues: List[ValidationIssue] = []
    
    # 1. Transaction Completeness
    if statement.transaction_count == 0:
        issues.append(
            ValidationIssue(
                issue_type="no_transactions",
                severity="WARNING",
                description="No transactions were extracted from the statement.",
            )
        )
    
    # 2. Layout Consistency
    if layout.detection_method == "none":
        issues.append(
            ValidationIssue(
                issue_type="no_layout_detected",
                severity="ERROR",
                description="Failed to detect or infer any table layout.",
            )
        )
    elif layout.detection_method == "data_inference":
        issues.append(
            ValidationIssue(
                issue_type="layout_inferred",
                severity="WARNING",
                description="Layout was inferred from data rows instead of a table header.",
            )
        )
        
    # 3. Numeric amount validity & Date validity
    # Count only rows that SHOULD have been assigned but weren't
    unassigned_count = 0
    missing_balance_count = 0
    has_balance_column = layout.balance_column is not None
    
    for row_result in assignment_report.rows:
        txn = row_result.transaction
        
        # Date validity: ParsedRow guarantees a date exists, so this is a rare case
        if not txn.date:
            issues.append(
                ValidationIssue(
                    issue_type="missing_date",
                    severity="ERROR",
                    description=f"Transaction at row {row_result.source_row_index} is missing a date.",
                )
            )
            
        # Numeric amount validity: only count as unassigned if we expected to assign
        # and failed. If layout has money columns but this row has no amounts in the
        # parsed amounts list, it's not an "unassigned" problem — it's a parse failure
        # (handled elsewhere). Only count it if we have parsed amounts but failed to
        # assign them to any semantic field.
        money_columns = [
            col for role in ["amount", "debit", "credit", "balance"]
            if getattr(layout, f"{role}_column", None) is not None
            for col in [getattr(layout, f"{role}_column")]
        ]
        
        has_any_amount_assigned = any([
            txn.amount is not None,
            txn.debit is not None,
            txn.credit is not None,
            txn.balance is not None
        ])
        
        if not has_any_amount_assigned and row_result.raw_amounts and money_columns:
            # Row had parsed amounts AND layout has money columns, but assignment failed
            unassigned_count += 1
                
        # Balance validity when balance exists in layout
        if has_balance_column and txn.balance is None and row_result.raw_amounts:
            missing_balance_count += 1

    if unassigned_count > 0:
        issues.append(
            ValidationIssue(
                issue_type="unassigned_amounts",
                severity="WARNING",
                description=f"{unassigned_count} transactions had amounts that could not be semantically assigned.",
            )
        )
        
    if has_balance_column and missing_balance_count > 0:
        issues.append(
            ValidationIssue(
                issue_type="missing_balances",
                severity="WARNING",
                description=f"{missing_balance_count} transactions are missing balance values despite layout having a balance column.",
            )
        )

    # 4. Header Extraction Completeness
    if not statement.metadata.account_number:
        issues.append(
            ValidationIssue(
                issue_type="missing_account_number",
                severity="WARNING",
                description="Could not extract account number from the statement header.",
            )
        )

    # Compute overall confidence
    confidence = _calculate_confidence(statement, layout, assignment_report, issues)
    
    # Determine pass/fail
    passed = not any(issue.severity == "ERROR" for issue in issues)
    
    report = ValidationReport(
        passed=passed,
        confidence=confidence,
        issues=issues,
    )
    
    logger.info(
        "Validation complete. Passed: %s | Confidence: %.2f | Issues: %d",
        report.passed,
        report.confidence,
        len(report.issues),
    )
    return report


def _calculate_confidence(
    statement: BankStatement,
    layout: LayoutDefinition,
    assignment_report: SemanticAssignmentReport,
    issues: List[ValidationIssue],
) -> float:
    """Calculate an overall 0.0-1.0 confidence score based on extraction signals."""
    if layout.detection_method == "none" or statement.transaction_count == 0:
        return 0.0
        
    base_score = 1.0
    
    # Penalize based on layout uncertainty
    if layout.detection_method == "data_inference":
        base_score -= 0.15
        
    layout_conf = layout.confidence_scores.get("table_header", 0.0)
    if layout_conf > 0 and layout_conf < 0.7:
        base_score -= 0.10
        
    # Penalize based on assignment success
    total_txns = statement.transaction_count
    if total_txns > 0:
        assigned_ratio = assignment_report.assigned_row_count / total_txns
        if assigned_ratio < 1.0:
            base_score -= (1.0 - assigned_ratio) * 0.4  # Max 40% penalty for unassigned rows
            
    # Penalize for other warnings/errors
    for issue in issues:
        if issue.severity == "ERROR":
            base_score -= 0.20
        elif issue.severity == "WARNING":
            base_score -= 0.05
            
    return max(0.0, min(1.0, round(base_score, 4)))