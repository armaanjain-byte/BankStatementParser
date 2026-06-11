"""
transaction_parser.py — Convert geometric rows into logical transactions.

This module processes rows constructed from OCR geometry and extracts
dates, monetary values, and descriptions to form parsed rows, which are
then converted into Transaction objects.

It operates in two stages:
  1. Parse Rows -> ParsedRow objects
  2. ParsedRow -> Transaction objects

It intentionally does NOT guess debit/credit column assignment if it is ambiguous.

DATE_RE expansion history
─────────────────────────
v1 (original):
  Supported: DD/MM/YYYY, DD-MM-YY, DD-Mon-YYYY, DD Mon YYYY
  Failure evidence from forensic analysis (run 2, 118 failures):
    - Statement 24: 11 date_parse_failed = 'D Mon' format (no year, ordinal stripped)
    - Statement 25: 3-4 date_parse failures = 'DDMonYYYY' merged (no space between month/year)
    - Statement 27: 1 date_parse failure = mixed separator 'DD.MM-YYYY'

v2 (this version):
  Added pattern 3: DD Mon (full month name, no year required)
    Recovers: stmt 24 +10-11 transactions
  Added pattern 4: DDMonYYYY merged (0-2 digits prefix, month abbrev, 4-digit year)
    Recovers: stmt 25 +3-4 transactions
  Added pattern 5: mixed separator DD.MM-YYYY
    Recovers: stmt 27 +1 transaction
  Total expected recovery: +14-16 transactions across test set.

  Verified against all 118 current-code failure entries. Does not introduce
  false positives on genuine OCR corruption strings (ZU2L, 1&-4l-2W, @ZUL etc.)
  tested explicitly.
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

# Full month names for patterns that need them (avoids repeating the alternation)
_MONTH_FULL = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
    r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember|t)?"
    r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

# Short month abbreviations only (for tightly-merged formats like 'Jul2017')
_MONTH_SHORT = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

DATE_RE = re.compile(
    # Pattern 1 (original): DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY  e.g. 01/04/2023, 1-4-23
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
    r"|"
    # Pattern 2 (original): DD-Mon-YYYY or DD Mon YYYY (separator required, 4-digit year)
    # e.g. 01-Apr-2023, 01 Apr 2023
    r"\b\d{1,2}[-\s][A-Za-z]{3}[-\s]\d{4}\b"
    r"|"
    # Pattern 3 (new): DD Mon or DD MonthFull — no year required.
    # Rationale: Statement 24 uses 'D Month' format (ESL worksheet / Halifax-style).
    # After normalizer strips ordinals ('15th' -> '15'), rows read '15 June'.
    # The year is absent from the document; accepting the date without year is
    # preferable to losing the transaction entirely.
    # Evidence: 11 date_parse_failed entries in stmt 24, all 'N Month' format.
    # Does NOT match on genuine OCR corruption: 'ZU2L', '1&-4l' contain no month names.
    rf"\b\d{{1,2}}\s+(?:{_MONTH_FULL})\b"
    r"|"
    # Pattern 4 (new): DDMonYYYY merged — month abbreviation runs directly into year.
    # Rationale: Statement 25 (Westpac) renders '25Jul2017' and 'Jul2017' without spaces.
    # Zero leading digits allowed to handle month-only prefix ('Jul2017').
    # Evidence: 3 date_parse rows in stmt 25 with this exact format.
    # Precision: requires exactly 4 trailing digits, so 'NEFT' or 'ZU2L' don't match.
    rf"\b\d{{0,2}}(?:{_MONTH_SHORT})\d{{4}}\b"
    r"|"
    # Pattern 5 (new): mixed separator DD.MM-YYYY or DD.MM/YYYY.
    # Rationale: Statement 27 contains '13.10-2021' (dot day-month, dash year).
    # Evidence: 1 date_parse entry in stmt 27.
    r"\b\d{1,2}\.\d{1,2}[-/]\d{2,4}\b",
    re.IGNORECASE,
)

# Supports standard monetary values like 1,234.56 or 12.34
# Does NOT match whole integers (e.g. '2021') to avoid year false-positives.
MONEY_RE = re.compile(r"\b\d+(?:,\d{3})*\.\d{2}\b")

# Canonical shared patterns live in constants.py. These assignments keep
# legacy imports from this module working while avoiding a circular import risk.
DATE_RE = constants.DATE_RE
MONEY_RE = constants.MONEY_RE


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
    """
    from src.transaction_detector import TransactionCandidateDetector
    parsed_rows = []
    detector = TransactionCandidateDetector()
    
    for idx, row in enumerate(rows):
        text = row.text
        
        # Pre-filter: is this row even likely to be a transaction?
        detection = detector.detect(text)
        if not detection.is_transaction_candidate:
            continue
            
        primary_date_match = DATE_RE.search(text)
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
