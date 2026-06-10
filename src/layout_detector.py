"""
layout_detector.py - Detect transaction table structure from OCR geometry.

This module identifies table headers and column layout before any debit,
credit, balance, or amount values are assigned to transactions.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.constants import ALIAS_TO_COLUMN, MONEY_RE
from src.schema import OCRToken
from src.transaction_reconstructor import Row, reconstruct_rows

logger = logging.getLogger(__name__)


TARGET_COLUMNS = ("date", "description", "debit", "credit", "balance", "amount")
MONEY_COLUMNS = frozenset({"debit", "credit", "balance", "amount"})


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


ALIAS_MAP: dict[str, tuple[str, float]] = {
    _normalize_text(alias): (canonical, confidence)
    for alias, (canonical, confidence) in ALIAS_TO_COLUMN.items()
}


HEADER_VALUE_DATE_RE = re.compile(
    r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$"
    r"|"
    r"^\d{1,2}\s+[A-Za-z]{3}\s*\d{0,4}$"
    r"|"
    r"^\d{1,2}[-\s][A-Za-z]{3}[-\s]?\d{2,4}$",
    re.IGNORECASE,
)

AMOUNT_LIKE_RE = re.compile(
    r"\b[$S]?\s*-?\d[\d,]*[.]\d{2}\b"
    r"|"
    r"\b[$S]?\s*-?\d[\d,]*[,]\d{2}\b",
    re.IGNORECASE,
)


@dataclass
class DetectedColumn:
    """A detected table column or header cell."""

    source_text: str
    normalized_name: Optional[str]
    x_min: int
    x_max: int
    x_center: float
    confidence: float
    source: str
    match_kind: str


@dataclass
class HeaderRowCandidate:
    """A row that may define transaction table columns."""

    source_row_index: int
    raw_text: str
    confidence: float
    detected_columns: List[DetectedColumn] = field(default_factory=list)


@dataclass
class LayoutDefinition:
    """Detected transaction table layout.

    The role fields identify which detected column should be treated as the
    date, description, debit, credit, balance, or generic amount column later.
    No transaction values are assigned here.
    """

    header_rows: List[HeaderRowCandidate] = field(default_factory=list)
    detected_columns: List[DetectedColumn] = field(default_factory=list)
    date_column: Optional[DetectedColumn] = None
    description_column: Optional[DetectedColumn] = None
    debit_column: Optional[DetectedColumn] = None
    credit_column: Optional[DetectedColumn] = None
    balance_column: Optional[DetectedColumn] = None
    amount_column: Optional[DetectedColumn] = None
    confidence_scores: dict[str, float] = field(default_factory=dict)
    detection_method: str = "none"

    def to_dict(self) -> dict:
        """Serialize layout definition to a JSON-ready dictionary."""
        return asdict(self)


def detect_layout(tokens: List[OCRToken], rows: List[Row]) -> LayoutDefinition:
    """Detect transaction table columns from OCR tokens and reconstructed rows."""
    del tokens  # Row tokens carry the geometry needed by the detector.

    header_candidates = _find_header_candidates(rows)
    if header_candidates:
        best = max(header_candidates, key=lambda candidate: candidate.confidence)
        layout = _layout_from_header(best, header_candidates)
        layout = _fill_missing_from_data_rows(layout, rows)
        logger.info(
            "Detected layout from header row %s with confidence %.3f",
            best.source_row_index,
            best.confidence,
        )
        return layout

    layout = _infer_layout_from_data_rows(rows)
    logger.info("Detected layout by data-row inference with method=%s", layout.detection_method)
    return layout


def save_layout_report(
    tokens: List[OCRToken],
    rows: List[Row],
    output_path: Union[str, Path],
    *,
    statement_id: Optional[str] = None,
) -> Path:
    """Detect layout and write a debugging report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layout = detect_layout(tokens, rows)
    payload = {
        "statement_id": statement_id,
        "ocr_token_count": len(tokens),
        "row_count": len(rows),
        "layout": layout.to_dict(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved layout report -> %s", output_path)
    return output_path


def load_ocr_tokens(input_path: Union[str, Path]) -> List[OCRToken]:
    """Load OCR tokens from an existing *_ocr.json file."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    return [
        OCRToken(
            text=t["text"],
            confidence=t["confidence"],
            bbox=tuple(t["bbox"]),
        )
        for t in data["tokens"]
    ]


def _find_header_candidates(rows: List[Row]) -> List[HeaderRowCandidate]:
    candidates: List[HeaderRowCandidate] = []

    for idx, row in enumerate(rows):
        columns = _detect_columns_in_header_row(row)
        roles = {c.normalized_name for c in columns if c.normalized_name}
        target_roles = roles.intersection(TARGET_COLUMNS)

        if "date" not in target_roles:
            continue
        if not target_roles.intersection({"description", "debit", "credit", "balance", "amount"}):
            continue

        confidence = _score_header_row(columns, target_roles)
        if confidence >= 0.40:
            candidates.append(
                HeaderRowCandidate(
                    source_row_index=idx,
                    raw_text=row.text,
                    confidence=round(confidence, 4),
                    detected_columns=columns,
                )
            )

    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def _detect_columns_in_header_row(row: Row) -> List[DetectedColumn]:
    columns: List[DetectedColumn] = []

    for token in row.tokens:
        normalized_name, match_confidence, match_kind = _match_column_name(token.text)
        if normalized_name:
            confidence = _combine_confidence(match_confidence, token.confidence)
            columns.append(
                DetectedColumn(
                    source_text=token.text,
                    normalized_name=normalized_name,
                    x_min=token.bbox[0],
                    x_max=token.bbox[2],
                    x_center=round((token.bbox[0] + token.bbox[2]) / 2.0, 1),
                    confidence=confidence,
                    source="header",
                    match_kind=match_kind,
                )
            )
        elif columns:
            columns.append(
                DetectedColumn(
                    source_text=token.text,
                    normalized_name=None,
                    x_min=token.bbox[0],
                    x_max=token.bbox[2],
                    x_center=round((token.bbox[0] + token.bbox[2]) / 2.0, 1),
                    confidence=round(token.confidence * 0.25, 4),
                    source="header",
                    match_kind="unknown",
                )
            )

    return _merge_duplicate_header_columns(columns)


def _layout_from_header(
    best: HeaderRowCandidate,
    all_candidates: List[HeaderRowCandidate],
) -> LayoutDefinition:
    columns = best.detected_columns
    role_columns = _choose_role_columns(columns)

    confidence_scores = {
        role: round(role_columns[role].confidence, 4) if role_columns.get(role) else 0.0
        for role in TARGET_COLUMNS
    }
    confidence_scores["table_header"] = best.confidence

    return LayoutDefinition(
        header_rows=all_candidates,
        detected_columns=list(columns),
        date_column=role_columns.get("date"),
        description_column=role_columns.get("description"),
        debit_column=role_columns.get("debit"),
        credit_column=role_columns.get("credit"),
        balance_column=role_columns.get("balance"),
        amount_column=role_columns.get("amount"),
        confidence_scores=confidence_scores,
        detection_method="header",
    )


def _fill_missing_from_data_rows(layout: LayoutDefinition, rows: List[Row]) -> LayoutDefinition:
    """Infer only clearly missing generic roles from data rows.

    This keeps header-derived debit/credit/balance decisions intact while
    allowing no-header fields such as a generic amount column to be reported
    when the data geometry is strong.
    """
    if layout.date_column and layout.description_column and layout.amount_column:
        return layout

    inferred = _infer_layout_from_data_rows(rows)
    if inferred.detection_method == "none":
        return layout

    for role in ("date", "description", "amount"):
        if role == "amount" and (layout.debit_column or layout.credit_column):
            continue
        attr = f"{role}_column"
        if getattr(layout, attr) is None and getattr(inferred, attr) is not None:
            setattr(layout, attr, getattr(inferred, attr))
            layout.confidence_scores[role] = getattr(inferred, attr).confidence

    if layout.detected_columns:
        existing = {(c.normalized_name, c.x_center) for c in layout.detected_columns}
    else:
        existing = set()
    for column in inferred.detected_columns:
        key = (column.normalized_name, column.x_center)
        if key not in existing:
            layout.detected_columns.append(column)

    return layout


def _infer_layout_from_data_rows(rows: List[Row]) -> LayoutDefinition:
    observations = []

    for row in rows:
        date_token = _first_date_value_token(row)
        amount_token = _rightmost_amount_token(row)
        if date_token and amount_token:
            description_token = _first_description_token(row, date_token, amount_token)
            observations.append((date_token, description_token, amount_token))

    if not observations:
        return LayoutDefinition(
            confidence_scores={role: 0.0 for role in TARGET_COLUMNS},
            detection_method="none",
        )

    confidence = min(0.70, 0.30 + len(observations) * 0.06)
    date_x = _average_center(t for t, _, _ in observations)
    amount_x = _average_center(t for _, _, t in observations)
    desc_tokens = [t for _, t, _ in observations if t is not None]
    description_x = _average_center(desc_tokens) if desc_tokens else (date_x + amount_x) / 2.0

    columns = [
        _inferred_column("date", date_x, confidence),
        _inferred_column("description", description_x, max(0.20, confidence - 0.08)),
        _inferred_column("amount", amount_x, confidence),
    ]
    role_columns = _choose_role_columns(columns)
    confidence_scores = {
        role: round(role_columns[role].confidence, 4) if role_columns.get(role) else 0.0
        for role in TARGET_COLUMNS
    }

    return LayoutDefinition(
        detected_columns=columns,
        date_column=role_columns.get("date"),
        description_column=role_columns.get("description"),
        amount_column=role_columns.get("amount"),
        confidence_scores=confidence_scores,
        detection_method="data_inference",
    )


def _choose_role_columns(columns: List[DetectedColumn]) -> dict[str, Optional[DetectedColumn]]:
    role_columns: dict[str, Optional[DetectedColumn]] = {role: None for role in TARGET_COLUMNS}

    for role in TARGET_COLUMNS:
        candidates = [c for c in columns if c.normalized_name == role]
        if candidates:
            role_columns[role] = max(candidates, key=lambda c: c.confidence)

    return role_columns


def _score_header_row(columns: List[DetectedColumn], target_roles: set[str]) -> float:
    matched = [c for c in columns if c.normalized_name in TARGET_COLUMNS]
    if not matched:
        return 0.0

    role_coverage = len(target_roles) / len(TARGET_COLUMNS)
    has_description = 0.10 if "description" in target_roles else 0.0
    has_money = 0.10 if target_roles.intersection(MONEY_COLUMNS) else 0.0
    confidence = sum(c.confidence for c in matched) / len(matched)

    return min(1.0, 0.25 + role_coverage * 0.45 + has_description + has_money + confidence * 0.25)


def _match_column_name(text: str) -> tuple[Optional[str], float, str]:
    normalized = _normalize_text(text)
    if not normalized:
        return None, 0.0, "none"

    if HEADER_VALUE_DATE_RE.fullmatch(text.strip()):
        return None, 0.0, "date_value"

    exact = ALIAS_MAP.get(normalized)
    if exact:
        return exact[0], exact[1], "exact_alias"

    words = normalized.split()
    for word in words:
        exact_word = ALIAS_MAP.get(word)
        if exact_word and len(word) > 2:
            return exact_word[0], exact_word[1] * 0.85, "word_alias"

    for alias, (canonical, confidence) in ALIAS_MAP.items():
        if canonical == "date":
            continue
        if len(alias) < 4:
            continue
        if alias in normalized:
            return canonical, confidence * 0.80, "contains_alias"

    best_alias = None
    best_ratio = 0.0
    for alias, (canonical, confidence) in ALIAS_MAP.items():
        if len(alias) < 5 or len(normalized) < 4:
            continue
        ratio = SequenceMatcher(None, normalized, alias).ratio()
        if ratio > best_ratio:
            best_alias = (canonical, confidence)
            best_ratio = ratio

    if best_alias and best_alias[0] != "date" and best_ratio >= 0.78:
        return best_alias[0], best_alias[1] * best_ratio * 0.75, "fuzzy_alias"

    return None, 0.0, "none"


def _merge_duplicate_header_columns(columns: List[DetectedColumn]) -> List[DetectedColumn]:
    merged: List[DetectedColumn] = []

    for column in columns:
        if (
            merged
            and column.normalized_name is not None
            and column.normalized_name == merged[-1].normalized_name
        ):
            previous = merged[-1]
            previous.source_text = f"{previous.source_text} {column.source_text}"
            previous.x_min = min(previous.x_min, column.x_min)
            previous.x_max = max(previous.x_max, column.x_max)
            previous.x_center = round((previous.x_min + previous.x_max) / 2.0, 1)
            previous.confidence = round(max(previous.confidence, column.confidence), 4)
        else:
            merged.append(column)

    return merged


def _first_date_value_token(row: Row) -> Optional[OCRToken]:
    for token in row.tokens[:3]:
        if HEADER_VALUE_DATE_RE.search(token.text.strip()):
            return token
    return None


def _rightmost_amount_token(row: Row) -> Optional[OCRToken]:
    amount_tokens = [
        token
        for token in row.tokens
        if MONEY_RE.search(token.text) or AMOUNT_LIKE_RE.search(token.text)
    ]
    if not amount_tokens:
        return None
    return max(amount_tokens, key=lambda token: token.bbox[2])


def _first_description_token(
    row: Row,
    date_token: OCRToken,
    amount_token: OCRToken,
) -> Optional[OCRToken]:
    for token in row.tokens:
        if token.bbox[0] <= date_token.bbox[2]:
            continue
        if token.bbox[0] >= amount_token.bbox[0]:
            continue
        if MONEY_RE.search(token.text) or AMOUNT_LIKE_RE.search(token.text):
            continue
        return token
    return None


def _inferred_column(normalized_name: str, x_center: float, confidence: float) -> DetectedColumn:
    width = 80 if normalized_name != "description" else 180
    return DetectedColumn(
        source_text=f"inferred_{normalized_name}",
        normalized_name=normalized_name,
        x_min=int(x_center - width / 2),
        x_max=int(x_center + width / 2),
        x_center=round(x_center, 1),
        confidence=round(confidence, 4),
        source="data_inference",
        match_kind="geometry",
    )


def _average_center(tokens: Iterable[OCRToken]) -> float:
    centers = [(token.bbox[0] + token.bbox[2]) / 2.0 for token in tokens]
    if not centers:
        return 0.0
    return sum(centers) / len(centers)


def _combine_confidence(match_confidence: float, ocr_confidence: float) -> float:
    return round(min(1.0, match_confidence * 0.65 + ocr_confidence * 0.35), 4)


def _default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem.replace('_ocr', '')}_layout_report.json"


def main() -> None:
    """CLI entry point for generating layout reports from OCR JSON files."""
    parser = argparse.ArgumentParser(
        description="Detect transaction table layout from OCR JSON.",
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
        if args.output:
            output_path = (
                args.output / _default_output_path(input_path).name
                if multiple or not args.output.suffix
                else args.output
            )
        else:
            output_path = _default_output_path(input_path)

        save_layout_report(
            tokens,
            rows,
            output_path,
            statement_id=input_path.stem.replace("_ocr", ""),
        )
        with open(output_path, encoding="utf-8") as f:
            combined_reports.append(json.load(f))

    if multiple and args.output:
        combined_path = args.output / "layout_report.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined_reports, f, indent=2, ensure_ascii=False)
        logger.info("Saved combined layout report -> %s", combined_path)


if __name__ == "__main__":
    main()
