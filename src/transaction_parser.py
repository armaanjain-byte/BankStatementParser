"""
transaction_parser.py — Convert geometric rows into logical transactions.

This module processes rows constructed from OCR geometry and extracts
dates, monetary values, and descriptions to form parsed rows, which are
then converted into Transaction objects.

It operates in two stages:
  1. Parse Rows -> ParsedRow objects
  2. ParsedRow -> Transaction objects

It intentionally does NOT guess debit/credit column assignment if it is ambiguous.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

from src.schema import Transaction
from src.transaction_reconstructor import Row

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
#  Extraction patterns
# ──────────────────────────────────────────────

# Supports: DD/MM/YYYY, DD/MM/YY, DD-MMM-YYYY, DD MMM YYYY (case-insensitive)
DATE_RE = re.compile(
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    r"|"
    r"\b\d{1,2}[-\s][A-Za-z]{3}[-\s]\d{4}\b",
    re.IGNORECASE
)

# Supports standard monetary values like 1,234.56 or 12.34
MONEY_RE = re.compile(r"\b\d+(?:,\d{3})*\.\d{2}\b")


# ──────────────────────────────────────────────
#  Stage 1 Logic
# ──────────────────────────────────────────────

def parse_rows(rows: List[Row]) -> List[ParsedRow]:
    """Identify transaction rows and extract core components.
    
    A transaction row typically contains:
      - A valid date
      - At least one monetary value
    """
    parsed_rows = []
    
    for idx, row in enumerate(rows):
        text = row.text
        
        primary_date_match = DATE_RE.search(text)
        amounts = MONEY_RE.findall(text)
        
        if primary_date_match and amounts:
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
            
    logger.info("Parsed %d transaction rows out of %d total rows", len(parsed_rows), len(rows))
    return parsed_rows


# ──────────────────────────────────────────────
#  Stage 2 Logic
# ──────────────────────────────────────────────

def build_transactions(parsed_rows: List[ParsedRow]) -> List[Transaction]:
    """Convert ParsedRow objects to schema Transaction objects.
    
    We preserve ambiguity: debit, credit, and balance are left as None 
    since we do not have sufficient information to map them yet without guessing.
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
    """Full pipeline: Row -> ParsedRow -> Transaction."""
    
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
