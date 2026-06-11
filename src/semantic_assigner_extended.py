"""
semantic_assigner_extended.py — Enhanced semantic assignment for complex layouts.

This module extends the semantic assigner to handle non-standard amount layouts:
  - 2 amounts: (Amount, Balance) or (Debit, Credit)
  - 3 amounts: (Debit, Credit, Balance) or (Amount, Reference, Balance)
  - 4+ amounts: Multiple currencies, fees, interest, etc.

The enhanced assignment logic:
  1. Detects number of money columns in layout
  2. Infers likely roles (Debit/Credit vs Amount, with Balance)
  3. Assigns amounts in left-to-right order
  4. Handles missing balance columns gracefully
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.layout_detector import LayoutDefinition, DetectedColumn
from src.transaction_parser import ParsedRow
from src.transaction_reconstructor import Row
from src.schema import Transaction

logger = logging.getLogger(__name__)


@dataclass
class LayoutInference:
    """Result of inferring layout from number and type of money columns."""
    num_amounts: int
    expected_layout: str  # "debit_credit_balance", "amount_balance", "amount_only", "complex"
    confidence: float
    inferred_roles: List[Tuple[str, int]]  # [(role_name, amount_index), ...]


def infer_layout_from_amounts(
    num_detected_money_columns: int,
    has_balance_column_in_layout: bool
) -> LayoutInference:
    """Infer likely layout from number of money columns.
    
    Args:
        num_detected_money_columns: Count of money columns detected in layout
        has_balance_column_in_layout: Whether layout explicitly has balance column
    
    Returns:
        LayoutInference with expected layout pattern and role assignments
    """
    if num_detected_money_columns == 0:
        return LayoutInference(
            num_amounts=0,
            expected_layout="no_amounts",
            confidence=0.0,
            inferred_roles=[]
        )
    
    elif num_detected_money_columns == 1:
        # Single amount column - could be Amount or Balance
        return LayoutInference(
            num_amounts=1,
            expected_layout="amount_only",
            confidence=0.8,
            inferred_roles=[("amount", 0)]
        )
    
    elif num_detected_money_columns == 2:
        # Two amount columns - could be:
        # (Amount, Balance) - most common for single-amount statements
        # (Debit, Credit) - for dual-amount statements
        roles = [("amount", 0), ("balance", 1)]
        if has_balance_column_in_layout:
            roles = [("amount", 0), ("balance", 1)]
        else:
            # No explicit balance - might be debit/credit
            roles = [("debit", 0), ("credit", 1)]
        
        return LayoutInference(
            num_amounts=2,
            expected_layout="amount_balance_or_debit_credit",
            confidence=0.7,
            inferred_roles=roles
        )
    
    elif num_detected_money_columns == 3:
        # Three amount columns - most likely:
        # (Debit, Credit, Balance) - standard layout
        # (Amount, something, Balance) - less common
        roles = [("debit", 0), ("credit", 1), ("balance", 2)]
        layout = "debit_credit_balance"
        
        if not has_balance_column_in_layout:
            # No balance column - could be (Amount, Reference, Fee) or similar
            roles = [("amount", 0), ("amount", 1), ("amount", 2)]
            layout = "multi_amount"
            confidence = 0.5
        else:
            confidence = 0.85
        
        return LayoutInference(
            num_amounts=3,
            expected_layout=layout,
            confidence=confidence,
            inferred_roles=roles
        )
    
    else:  # 4+ columns
        # Complex layout - assign in order as debit, credit, balance, then remaining
        roles = [("debit", 0), ("credit", 1), ("balance", 2)]
        for i in range(3, num_detected_money_columns):
            roles.append((f"amount_{i}", i))
        
        return LayoutInference(
            num_amounts=num_detected_money_columns,
            expected_layout="complex_multi_amount",
            confidence=0.4,
            inferred_roles=roles
        )


def assign_amounts_by_inferred_layout(
    parsed_row: ParsedRow,
    inferred_layout: LayoutInference
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Assign parsed amounts to debit/credit/balance using inferred layout.
    
    Args:
        parsed_row: The ParsedRow with extracted amounts
        inferred_layout: The inferred layout structure
    
    Returns:
        Tuple of (debit, credit, balance, reason_string)
    """
    debit = None
    credit = None
    balance = None
    
    amounts = [_parse_amount(a) for a in parsed_row.amounts]
    
    if not amounts or all(a is None for a in amounts):
        return None, None, None, "No parseable amounts in row"
    
    # Assign based on inferred roles
    for role, amount_idx in inferred_layout.inferred_roles:
        if amount_idx >= len(amounts):
            break
        
        amount_val = amounts[amount_idx]
        if amount_val is None:
            continue
        
        if role == "debit":
            debit = amount_val
        elif role == "credit":
            credit = amount_val
        elif role == "balance":
            balance = amount_val
        elif role == "amount":
            # Single amount column - assign as debit if first, credit if context suggests
            if debit is None:
                debit = amount_val
            elif credit is None:
                credit = amount_val
            else:
                # Already have both, use as balance
                if balance is None:
                    balance = amount_val
    
    if debit or credit or balance:
        return debit, credit, balance, f"Assigned via inferred {inferred_layout.expected_layout} layout"
    else:
        return None, None, None, "Could not assign amounts using inferred layout"


def assign_with_layout_inference(
    parsed_row: ParsedRow,
    row: Row,
    layout: LayoutDefinition
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Enhanced assignment using layout inference.
    
    This handles non-standard layouts by:
    1. Counting money columns detected in layout
    2. Inferring expected role assignments (debit/credit vs amount, with balance)
    3. Assigning amounts to roles in left-to-right order
    4. Falling back to positional assignment if inference fails
    
    Args:
        parsed_row: The ParsedRow with extracted amounts
        row: The original Row (for geometric fallback)
        layout: The detected/inferred LayoutDefinition
    
    Returns:
        Tuple of (debit, credit, balance, reason_string)
    """
    # Count money columns
    money_columns = []
    for role in ["debit", "credit", "balance", "amount"]:
        col = getattr(layout, f"{role}_column")
        if col is not None:
            money_columns.append((role, col))
    
    # Infer layout
    num_columns = len(money_columns)
    has_balance = layout.balance_column is not None
    
    inferred = infer_layout_from_amounts(num_columns, has_balance)
    logger.debug(
        "Inferred layout: %s (confidence=%.2f) for %d money columns",
        inferred.expected_layout,
        inferred.confidence,
        num_columns
    )
    
    # Try assignment with inferred layout
    debit, credit, balance, reason = assign_amounts_by_inferred_layout(parsed_row, inferred)
    
    if debit or credit or balance:
        return debit, credit, balance, reason
    
    # Fallback: if inference didn't work, return no assignment but explain why
    return None, None, None, f"Inference failed for {inferred.expected_layout} layout"


def _parse_amount(raw: str) -> Optional[float]:
    """Parse a monetary amount string."""
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None