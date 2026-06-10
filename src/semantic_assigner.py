"""
semantic_assigner.py - Map parsed monetary values to semantic fields.

This module consumes ParsedRow objects and a LayoutDefinition. It assigns
amount, debit, credit, and balance values only when the detected layout makes
the mapping explicit. It does not infer transaction direction, validate
balances, or repair OCR.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.layout_detector import (
    DetectedColumn,
    LayoutDefinition,
    detect_layout,
    load_ocr_tokens,
)
from src.schema import Transaction
from src.transaction_parser import ParsedRow, parse_rows
from src.transaction_reconstructor import Row, reconstruct_rows

logger = logging.getLogger(__name__)


MONEY_ROLES = ("debit", "credit", "balance", "amount")


@dataclass
class AssignmentResult:
    """Debuggable result of assigning one ParsedRow."""

    source_row_index: int
    raw_text: str
    raw_amounts: List[str]
    layout_used: dict
    assigned_values: dict[str, Optional[float]]
    confidence: float
    reason: str
    transaction: Transaction

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["transaction"] = asdict(self.transaction)
        return payload


@dataclass
class SemanticAssignmentReport:
    """Report for all parsed rows in one statement."""

    statement_id: Optional[str]
    layout_summary: dict
    assignment_count: int
    assigned_row_count: int
    rows: List[AssignmentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "statement_id": self.statement_id,
            "layout_summary": self.layout_summary,
            "assignment_count": self.assignment_count,
            "assigned_row_count": self.assigned_row_count,
            "rows": [row.to_dict() for row in self.rows],
        }


def assign_transaction(parsed_row: ParsedRow, row: Row, layout: LayoutDefinition) -> Transaction:
    """Assign semantic fields for one ParsedRow and return a Transaction."""
    return assign_semantics(parsed_row, row, layout).transaction


def assign_semantics(parsed_row: ParsedRow, row: Row, layout: LayoutDefinition) -> AssignmentResult:
    """Assign one ParsedRow using the detected table layout.

    Values are assigned by left-to-right order of detected money columns.
    If the parsed amount count does not exactly match the layout columns,
    a geometric nearest-column fallback is used.
    """
    money_columns = _layout_money_columns(layout)
    amount_values = [_parse_amount(value) for value in parsed_row.amounts]

    assigned_values = {
        "amount": None,
        "debit": None,
        "credit": None,
        "balance": None,
    }
    reason = "no money columns detected in layout"
    confidence = 0.0

    if money_columns:
        if len(amount_values) == len(money_columns) and all(value is not None for value in amount_values):
            for (role, _column), value in zip(money_columns, amount_values):
                assigned_values[role] = value
            reason = "assigned by detected layout column order"
            confidence = _assignment_confidence(money_columns)
        elif len(amount_values) < len(money_columns) and all(value is not None for value in amount_values):
            amount_centers = [_find_amount_x_center(amt_str, row) for amt_str in parsed_row.amounts]
            if all(c is not None for c in amount_centers):
                temp_assignments = []
                for amt_val, center in zip(amount_values, amount_centers):
                    nearest_role = min(money_columns, key=lambda mc: abs(center - mc[1].x_center))[0]
                    temp_assignments.append((amt_val, nearest_role))
                
                roles_chosen = [role for _, role in temp_assignments]
                if len(set(roles_chosen)) == len(temp_assignments):
                    for amt_val, role in temp_assignments:
                        assigned_values[role] = amt_val
                    reason = "assigned by geometric nearest-column fallback"
                    confidence = _assignment_confidence(money_columns) * 0.9
                else:
                    reason = f"geometric fallback resulted in conflicts: {roles_chosen}"
            else:
                reason = "could not locate all amount tokens for geometric fallback"
        elif any(value is None for value in amount_values):
            reason = "one or more parsed amounts could not be converted"
        else:
            reason = (
                "amount count does not match detected money columns "
                f"({len(amount_values)} amounts, {len(money_columns)} columns)"
            )

    transaction = Transaction(
        date=parsed_row.date,
        description=parsed_row.description,
        amount=assigned_values["amount"],
        debit=assigned_values["debit"],
        credit=assigned_values["credit"],
        balance=assigned_values["balance"],
    )

    return AssignmentResult(
        source_row_index=parsed_row.source_row_index,
        raw_text=parsed_row.raw_text,
        raw_amounts=parsed_row.amounts,
        layout_used=_layout_used_payload(layout, money_columns),
        assigned_values=assigned_values,
        confidence=round(confidence, 4),
        reason=reason,
        transaction=transaction,
    )


def assign_rows(parsed_rows: List[ParsedRow], rows: List[Row], layout: LayoutDefinition) -> List[AssignmentResult]:
    """Assign all parsed rows using one layout definition."""
    return [assign_semantics(parsed_row, rows[parsed_row.source_row_index], layout) for parsed_row in parsed_rows]


def build_semantic_assignment_report(
    parsed_rows: List[ParsedRow],
    rows: List[Row],
    layout: LayoutDefinition,
    *,
    statement_id: Optional[str] = None,
) -> SemanticAssignmentReport:
    """Build a debugging report for semantic assignment."""
    results = assign_rows(parsed_rows, rows, layout)
    assigned_row_count = sum(
        1
        for row in results
        if any(value is not None for value in row.assigned_values.values())
    )

    return SemanticAssignmentReport(
        statement_id=statement_id,
        layout_summary=_layout_summary(layout),
        assignment_count=len(results),
        assigned_row_count=assigned_row_count,
        rows=results,
    )


def save_semantic_assignment_report(
    parsed_rows: List[ParsedRow],
    rows: List[Row],
    layout: LayoutDefinition,
    output_path: Union[str, Path],
    *,
    statement_id: Optional[str] = None,
) -> Path:
    """Write semantic assignment debug output to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_semantic_assignment_report(
        parsed_rows,
        rows,
        layout,
        statement_id=statement_id,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved semantic assignment report -> %s", output_path)
    return output_path


def run_from_ocr_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """Run reconstruction, layout detection, parsing, and semantic assignment."""
    input_path = Path(input_path)
    tokens = load_ocr_tokens(input_path)
    rows = reconstruct_rows(tokens)
    layout = detect_layout(tokens, rows)
    parsed_rows = parse_rows(rows)

    return save_semantic_assignment_report(
        parsed_rows,
        rows,
        layout,
        output_path,
        statement_id=input_path.stem.replace("_ocr", ""),
    )


def _find_amount_x_center(amount_str: str, row: Row) -> Optional[float]:
    """Find the geometric x-center of an amount string within a row."""
    start_idx = row.text.find(amount_str)
    if start_idx == -1:
        return None
    end_idx = start_idx + len(amount_str)
    
    current_idx = 0
    matched_tokens = []
    for token in row.tokens:
        token_start = current_idx
        token_end = current_idx + len(token.text)
        
        # Check if the token overlaps with the amount string
        if token_end > start_idx and token_start < end_idx:
            matched_tokens.append(token)
            
        current_idx += len(token.text) + 1  # +1 for the space separator
        
    if matched_tokens:
        x_min = min(t.bbox[0] for t in matched_tokens)
        x_max = max(t.bbox[2] for t in matched_tokens)
        return (x_min + x_max) / 2.0
        
    return None


def _layout_money_columns(layout: LayoutDefinition) -> List[tuple[str, DetectedColumn]]:
    columns: List[tuple[str, DetectedColumn]] = []
    for role in MONEY_ROLES:
        column = getattr(layout, f"{role}_column")
        if column is not None:
            columns.append((role, column))

    return sorted(columns, key=lambda item: item[1].x_center)


def _parse_amount(raw: str) -> Optional[float]:
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _assignment_confidence(money_columns: List[tuple[str, DetectedColumn]]) -> float:
    if not money_columns:
        return 0.0
    return sum(column.confidence for _, column in money_columns) / len(money_columns)


def _layout_used_payload(
    layout: LayoutDefinition,
    money_columns: List[tuple[str, DetectedColumn]],
) -> dict:
    return {
        "detection_method": layout.detection_method,
        "table_header_confidence": layout.confidence_scores.get("table_header", 0.0),
        "money_columns": [
            {
                "role": role,
                "source_text": column.source_text,
                "x_center": column.x_center,
                "confidence": column.confidence,
            }
            for role, column in money_columns
        ],
    }


def _layout_summary(layout: LayoutDefinition) -> dict:
    summary = {
        "detection_method": layout.detection_method,
        "confidence_scores": layout.confidence_scores,
        "columns": {},
    }
    for role in ("date", "description", "debit", "credit", "balance", "amount"):
        column = getattr(layout, f"{role}_column")
        summary["columns"][role] = _column_payload(column)
    return summary


def _column_payload(column: Optional[DetectedColumn]) -> Optional[dict]:
    if column is None:
        return None
    return {
        "source_text": column.source_text,
        "normalized_name": column.normalized_name,
        "x_min": column.x_min,
        "x_max": column.x_max,
        "x_center": column.x_center,
        "confidence": column.confidence,
        "source": column.source,
        "match_kind": column.match_kind,
    }


def _default_output_path(input_path: Path) -> Path:
    statement_id = input_path.stem.replace("_ocr", "")
    return input_path.parent / f"{statement_id}_semantic_assignment_report.json"


def main() -> None:
    """CLI entry point for semantic assignment reports."""
    parser = argparse.ArgumentParser(
        description="Assign parsed transaction values using detected layout.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more *_ocr.json files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for one input, or output directory for multiple inputs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    multiple = len(args.inputs) > 1
    if multiple and args.output and args.output.suffix:
        parser.error("--output must be a directory when multiple inputs are provided")

    combined_reports = []
    for input_path in args.inputs:
        tokens = load_ocr_tokens(input_path)
        rows = reconstruct_rows(tokens)
        layout = detect_layout(tokens, rows)
        parsed_rows = parse_rows(rows)

        if args.output:
            output_path = (
                args.output / _default_output_path(input_path).name
                if multiple or not args.output.suffix
                else args.output
            )
        else:
            output_path = _default_output_path(input_path)

        save_semantic_assignment_report(
            parsed_rows,
            rows,
            layout,
            output_path,
            statement_id=input_path.stem.replace("_ocr", ""),
        )
        with open(output_path, encoding="utf-8") as f:
            combined_reports.append(json.load(f))

    if multiple and args.output:
        combined_path = args.output / "semantic_assignment_report.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined_reports, f, indent=2, ensure_ascii=False)
        logger.info("Saved combined semantic assignment report -> %s", combined_path)


if __name__ == "__main__":
    main()
