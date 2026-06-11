"""
row_classifier.py — Aggressive pre-filtering to identify transaction rows.

This module classifies rows BEFORE date extraction to eliminate false positives
from headers, footers, page markers, and other non-transaction content.

Classification hierarchy:
  1. Header rows (e.g., "Statement for Period", "Account Number")
  2. Footer/summary rows (e.g., "Total Balance", "End of Statement")
  3. Page markers (e.g., "Page 1 of 5")
  4. Bank/account metadata (e.g., "ICICI Bank", "Savings Account")
  5. Reward/summary sections (e.g., "Reward Points", "Account Summary")
  6. Disclaimers/terms (e.g., "Terms and Conditions")
  7. Transaction row (everything else that might be a transaction)

Rows classified as non-transaction are REJECTED before date parsing,
eliminating the 333+ "no_date_detected" false positives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RowClassification:
    """Result of classifying a single row."""
    is_transaction_candidate: bool
    classification: str  # "header", "footer", "page_marker", "metadata", "summary", "disclaimer", "transaction"
    confidence: float
    reason: str


class RowClassifier:
    """Aggressive classifier to identify transaction rows and reject obvious non-transactions."""
    
    def __init__(self):
        # Header patterns - appear at start of statement
        self.header_patterns = re.compile(
            r"\b(statement\s+(for|of|period|date|from)|"
            r"account\s+(number|holder|no|name|type)|"
            r"customer\s+(name|id|code)|"
            r"branch\s+(code|name)|"
            r"ifsc|swift|sort code|micr|iban|account\s+type|"
            r"opening\s+balance|start\s+balance|opening\s+date|"
            r"dear\s+customer|valued\s+customer)\b",
            re.IGNORECASE
        )
        
        # Footer patterns - appear at end of statement
        self.footer_patterns = re.compile(
            r"\b(closing\s+balance|end\s+balance|final\s+balance|"
            r"brought\s+forward|carried\s+forward|total\s+(balance|debit|credit|amount)|"
            r"account\s+summary|statement\s+summary|balance\s+summary|"
            r"end\s+of\s+statement|end\s+of\s+transactions?|"
            r"thankyou|thank\s+you|contact\s+us|customer\s+service|"
            r"available\s+balance|minimum\s+due|overdraft|reward|loyalty)\b",
            re.IGNORECASE
        )
        
        # Page markers - indicate pagination
        self.page_patterns = re.compile(
            r"\b(page\s+\d+|of\s+\d+|(?:^|\s)-\s*\d+\s*-(?:\s|$)|continued|contd|contd\.|more|next\s+page|"
            r"^page\b|^\d+\s*$)\b",
            re.IGNORECASE
        )
        
        # Bank/account metadata
        self.metadata_patterns = re.compile(
            r"\b(hdfc|icici|axis|sbi|pnb|bob|canara|union|kotak|induslnd|federal|"
            r"hsbc|citi|dbs|scb|bank|savings\s+account|current\s+account|"
            r"credit\s+card|debit\s+card|visa|mastercard|amex)\b",
            re.IGNORECASE
        )
        
        # Reward/summary sections
        self.summary_patterns = re.compile(
            r"\b(reward\s+points|loyalty\s+points|cashback|bonus|points\s+summary|"
            r"transaction\s+summary|account\s+summary|period\s+summary|"
            r"tax\s+information|interest\s+earned|charges|fees)\b",
            re.IGNORECASE
        )
        
        # Disclaimer/terms/conditions
        self.disclaimer_patterns = re.compile(
            r"\b(terms\s+and\s+conditions|disclaimers?|important|please\s+read|"
            r"note\s+:|condition|limitation|restrictions?|"
            r"regulatory|compliance|legal|liability|generated|report|"
            r"confidential|copyright|®|™)\b",
            re.IGNORECASE
        )
        
        # Column header row patterns
        self.column_header_patterns = re.compile(
            r"\b(date|narration|description|particulars|debit|credit|"
            r"withdrawal|deposit|balance|amount|txn\s+date|posting\s+date|"
            r"value\s+date|reference|remarks|cheque|chq|instrument|type)\b",
            re.IGNORECASE | re.MULTILINE
        )
        
        # Patterns that might appear in row separators or formatting
        self.separator_patterns = re.compile(
            r"^[\s\-_=*~^]+$|^-{3,}$|^_{3,}$|^={3,}$"
        )

    def classify(self, row_text: str) -> RowClassification:
        """Classify a single row.
        
        Args:
            row_text: The text of the row to classify.
        
        Returns:
            RowClassification with decision and reasoning.
        """
        text = row_text.strip()
        
        # Empty rows are not transactions
        if not text or len(text) < 3:
            return RowClassification(
                is_transaction_candidate=False,
                classification="empty",
                confidence=1.0,
                reason="Row is empty or too short"
            )
        
        # Separator rows
        if self.separator_patterns.match(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="separator",
                confidence=1.0,
                reason="Row is a visual separator"
            )
        
        # Header rows
        if self.header_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="header",
                confidence=0.95,
                reason="Row matches header pattern"
            )
        
        # Footer rows
        if self.footer_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="footer",
                confidence=0.95,
                reason="Row matches footer pattern"
            )
        
        # Page markers
        if self.page_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="page_marker",
                confidence=0.92,
                reason="Row appears to be a page marker"
            )
        
        # Bank/account metadata
        if self.metadata_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="metadata",
                confidence=0.90,
                reason="Row contains bank or account metadata"
            )
        
        # Reward/summary sections
        if self.summary_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="summary",
                confidence=0.90,
                reason="Row appears to be a summary or reward section"
            )
        
        # Disclaimer/terms
        if self.disclaimer_patterns.search(text):
            return RowClassification(
                is_transaction_candidate=False,
                classification="disclaimer",
                confidence=0.85,
                reason="Row appears to be disclaimer or terms"
            )
        
        # Column header rows (multiple column names in single row)
        column_matches = len(self.column_header_patterns.findall(text))
        if column_matches >= 3:  # Multiple column headers
            return RowClassification(
                is_transaction_candidate=False,
                classification="column_header",
                confidence=0.95,
                reason=f"Row contains {column_matches} column header keywords"
            )
        
        # If we reach here, it might be a transaction row
        return RowClassification(
            is_transaction_candidate=True,
            classification="transaction",
            confidence=0.6,
            reason="Row did not match any non-transaction pattern"
        )


def classify_rows(rows: list[str]) -> tuple[list[str], dict[str, int]]:
    """Filter rows to only transaction candidates.
    
    Args:
        rows: List of row text strings.
    
    Returns:
        Tuple of (filtered_rows, classification_counts)
        where classification_counts maps classification type to count.
    """
    classifier = RowClassifier()
    filtered_rows = []
    classification_counts = {}
    
    for row in rows:
        classification = classifier.classify(row)
        
        # Count classification
        classification_type = classification.classification
        classification_counts[classification_type] = classification_counts.get(classification_type, 0) + 1
        
        # Keep transaction candidates
        if classification.is_transaction_candidate:
            filtered_rows.append(row)
        else:
            logger.debug(
                "Filtered non-transaction row [%s]: %s",
                classification_type,
                row[:60]
            )
    
    logger.info(
        "Row classification: %d/%d rows kept (transaction candidates)",
        len(filtered_rows),
        len(rows)
    )
    for class_type, count in sorted(classification_counts.items()):
        if count > 0:
            logger.debug("  %s: %d rows", class_type, count)
    
    return filtered_rows, classification_counts