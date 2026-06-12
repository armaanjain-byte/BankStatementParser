"""
transaction_parser.py — Convert geometric rows into logical transactions.

This module processes rows constructed from OCR geometry and extracts
dates, monetary values, and descriptions to form parsed rows, which are
then converted into Transaction objects.

It operates in two stages:
  1. Parse Rows -> ParsedRow objects
  2. ParsedRow -> Transaction objects

It intentionally does NOT guess debit/credit column assignment if it is ambiguous.

DATE_RE Consolidation
─────────────────────
This module now imports DATE_RE from constants.py (the comprehensive pattern
with 8 detection formats), not a local version. This ensures all modules
(transaction_parser, evaluation, transaction_detector) use the same regex,
eliminating date detection mismatches.

Previously, each module had its own DATE_RE, causing critical failures:
- evaluation.py used 5 patterns → missed valid dates
- constants.py had 8 patterns → caught more dates
- This mismatch meant evaluation.py reported higher failure rates than actual

Fix: All modules now import from constants.DATE_RE.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

from src import constants
from src.schema import Transaction
from src.transaction_reconstructor import Row

# Import the consolidated DATE_RE from constants (8 patterns)
# This replaces the previous local 5-pattern version
DATE_RE = constants.DATE_RE
MONEY_RE = constants.MONEY_RE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Stage 1: Intermediate parsed representation
# ──────────────────────────────────────────────

@dataclass
class ParsedRow:
    """An intermediate representation of a row before final Transaction conversion.
    
    Attributes:
        raw_text:         The original concatenated text of the row.
        date:             The extracted date string.
        amounts:          All extracted monetary values as strings.
        description:      The remaining text after removing date and amounts.
        source_row_index: The 0-based index of the source row from the reconstruction.
    """
    raw_text: str
    date: str
    amounts: List[str]
    description: str
    source_row_index: int


# ──────────────────────────────────────────────
#  Stage 1 Logic
# ──────────────────────────────────────────────

HEADER_FOOTER_KEYWORDS = frozenset({
    "statement of", "statement for", "account no", "account number",
    "for the period", "opening balance", "closing balance", "start balance",
    "end balance", "brought forward", "carried forward", "page", "total",
    "account summary", "sort code", "branch", "ifsc", "swift",
})

def _is_header_or_footer(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in HEADER_FOOTER_KEYWORDS)


def parse_rows(rows: List[Row]) -> List[ParsedRow]:
    """Identify transaction rows and extract core components.
    
    A transaction row typically contains:
      - A valid date (matched by DATE_RE)
      - At least one monetary value (matched by MONEY_RE)
    
    Rows that contain a date but no monetary value are excluded here.
    They are categorised as 'insufficient_numeric_values' by the evaluation
    framework, which is accurate — the row split during reconstruction.
    
    NOTE: ISO dates like "2019-01-25" are now properly recognized by the
    consolidated DATE_RE pattern (Pattern 6), so they should no longer be
    rejected as "insufficient_numeric_values".
    """
    from src.transaction_detector import TransactionCandidateDetector
    from src.row_classifier import RowClassifier
    
    parsed_rows = []
    detector = TransactionCandidateDetector()
    classifier = RowClassifier()
    
    for idx, row in enumerate(rows):
        text = row.text
        
        # Pre-filter 1: Is this row even a non-header/footer/page-marker?
        classification = classifier.classify(text)
        if not classification.is_transaction_candidate:
            logger.debug("Skipped non-transaction row [%s]: %s", 
                        classification.classification, text[:60])
            continue
        
        # Pre-filter 2: Is this row likely to be a transaction?
        detection = detector.detect(text)
        if not detection.is_transaction_candidate:
            continue
            
        # Find the FIRST VALID date candidate rather than only testing the
        # first match. An OCR-mangled amount that happens to match DATE_RE
        # and fails validation must not discard a row that also contains a
        # genuinely valid date later in the text.
        from src.date_validator import validate_date_candidate
        primary_date_match = None
        for candidate in DATE_RE.finditer(text):
            is_valid, _ = validate_date_candidate(candidate.group(0))
            if is_valid:
                primary_date_match = candidate
                break
                
        amounts = MONEY_RE.findall(text)
        
        if primary_date_match and amounts:
            if _is_header_or_footer(text):
                logger.debug("Skipped header/footer row: %s", text[:60])
                continue

            # We found a transaction row
            date_str = primary_date_match.group(0)
            
            # Remove only the transaction date. Later date-like text can be
            # meaningful narration, such as interest-period ranges.
            description = (
                text[:primary_date_match.start()]
                + text[primary_date_match.end():]
            )
            for a in amounts:
                description = description.replace(a, "", 1)
                
            # Normalize whitespace
            description = re.sub(r"\s+", " ", description).strip()
            
            parsed_rows.append(ParsedRow(
                raw_text=text,
                date=date_str,
                amounts=amounts,
                description=description,
                source_row_index=idx
            ))
            
    # -- Contextual Year Inference --
    import datetime
    from collections import Counter
    
    current_year = datetime.datetime.now().year
    valid_years = []
    
    # First pass: collect all valid years
    for pr in parsed_rows:
        try:
            # All canonicalized dates are in YYYY-MM-DD format
            if len(pr.date) >= 10 and pr.date[4] == '-' and pr.date[7] == '-':
                y = int(pr.date[:4])
                if 1990 <= y <= current_year + 1:
                    valid_years.append(y)
        except ValueError:
            pass
            
    if valid_years:
        dominant_year = Counter(valid_years).most_common(1)[0][0]
        
        # Second pass: replace out-of-bounds years with the dominant year
        for pr in parsed_rows:
            try:
                if len(pr.date) >= 10 and pr.date[4] == '-' and pr.date[7] == '-':
                    y = int(pr.date[:4])
                    if y < 1990 or y > current_year + 1:
                        # Replace the YYYY part with dominant_year
                        pr.date = f"{dominant_year:04d}{pr.date[4:]}"
                        logger.debug("Repaired out-of-range year %d to dominant year %d for row %d", y, dominant_year, pr.source_row_index)
            except ValueError:
                pass
                
    logger.info("Parsed %d transaction rows out of %d total rows", len(parsed_rows), len(rows))
    return parsed_rows


# ──────────────────────────────────────────────
#  Stage 2 Logic
# ──────────────────────────────────────────────

def build_transactions(parsed_rows: List[ParsedRow]) -> List[Transaction]:
    """Convert ParsedRow objects to schema Transaction objects.
    
    We preserve ambiguity: debit, credit, and balance are left as None 
    since we do not have sufficient information to map them yet without guessing.
    Semantic assignment (src.semantic_assigner) handles the mapping.
    """
    transactions = []
    
    for pr in parsed_rows:
        txn = Transaction(
            date=pr.date,
            description=pr.description,
            debit=None,
            credit=None,
            balance=None
        )
        transactions.append(txn)
        
    logger.info("Built %d Transaction objects", len(transactions))
    return transactions


# ──────────────────────────────────────────────
#  Public API / Workflow
# ──────────────────────────────────────────────

def process_reconstructed_rows(
    rows: List[Row], 
    debug_output_path: Union[str, Path, None] = None
) -> List[Transaction]:
    """Full pipeline: Row -> ParsedRow -> Transaction.
    
    Note: This function produces Transaction objects with debit/credit/balance
    set to None. For fully assigned transactions, use semantic_assigner.assign_rows
    with a detected LayoutDefinition. This function is provided for debugging
    and standalone testing only.
    """
    
    # Stage 1
    parsed_rows = parse_rows(rows)
    
    # Save debug output if requested
    if debug_output_path:
        debug_output_path = Path(debug_output_path)
        debug_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        debug_payload = [asdict(pr) for pr in parsed_rows]
        with open(debug_output_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved parsed rows debug output to %s", debug_output_path)
        
    # Stage 2
    transactions = build_transactions(parsed_rows)
    
    return transactions


if __name__ == "__main__":
    # Quick standalone test against an existing rows JSON
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.transaction_parser outputs/<file>_rows.json")
        sys.exit(1)
        
    input_path = Path(sys.argv[1])
    
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
        
    # Rehydrate rows
    # Note: we don't fully need the OCRToken objects inside Row for parsing text
    loaded_rows = [
        Row(y_center=r["y_center"], tokens=[], text=r["text"]) 
        for r in data["rows"]
    ]
    
    out_json = input_path.parent / "parsed_rows.json"
    txns = process_reconstructed_rows(loaded_rows, debug_output_path=out_json)
    
    print(f"\nExtracted {len(txns)} transactions.")
    for t in txns[:5]:
        print(f"  {t.date} | {t.description[:40]:<40} | D:{t.debit} C:{t.credit} B:{t.balance}")
    if len(txns) > 5:
        print("  ...")