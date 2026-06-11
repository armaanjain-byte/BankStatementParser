"""
transaction_detector.py — Heuristic-based transaction candidate detection.

Provides a fast pre-filter to determine if a reconstructed row is likely to be a
transaction before passing it to the rigorous parsing pipeline.
"""

import re
from dataclasses import dataclass
from typing import List

from src.constants import DATE_RE, MONEY_RE


@dataclass
class DetectionResult:
    """Result of transaction candidate detection."""
    is_transaction_candidate: bool
    confidence_score: float
    reasons: List[str]


class TransactionCandidateDetector:
    """Scores a row text based on positive and negative transaction signals."""

    def __init__(self):
        # High-signal keywords often found in transaction narrations
        self.transaction_keywords = re.compile(
            r"\b(ATM|POS|UPI|IMPS|NEFT|RTGS|TRANSFER|PAYMENT|PURCHASE|DEPOSIT|"
            r"WITHDRAWAL|CHEQUE|BACS|CARD|INTEREST|SALARY)\b", 
            re.IGNORECASE
        )
        
        # High-signal negative keywords usually found in headers/footers
        self.negative_keywords = re.compile(
            r"\b(Page|Statement|Account Number|Customer Name|Balance Summary|"
            r"Reward Points|MICR|IFSC|Branch|End of Statement|Terms and Conditions)\b",
            re.IGNORECASE
        )
        
        # Broad pattern to catch mangled OCR dates that DATE_RE might reject
        self.date_candidate_re = re.compile(
            r"\b\d{1,2}[\s/\-\.]+(?:\d{1,2}|[A-Za-z]{3})[\s/\-\.]+(?:\d{2,4})\b|"
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{2,4}\b|"
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\b|"
            r"\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b|"
            r"\b\d{1,2}[A-Za-z]{3}\d{0,4}\b",
            re.IGNORECASE
        )
        
        self.cr_dr_re = re.compile(r"\b(Cr|Dr|CR|DR)\b")

    def detect(self, row_text: str) -> DetectionResult:
        """Analyze row text and return a detection result."""
        reasons = []
        score = 0.0
        
        # 1. Negative signals (instant rejection or heavy penalty)
        if self.negative_keywords.search(row_text):
            return DetectionResult(False, 0.0, ["Contains negative signal keywords"])
            
        # 2. Positive signals
        if DATE_RE.search(row_text) or self.date_candidate_re.search(row_text):
            score += 0.4
            reasons.append("Contains recognizable date")
            
        amounts = MONEY_RE.findall(row_text)
        if len(amounts) >= 1:
            score += 0.3
            reasons.append("Contains monetary values")
        if len(amounts) >= 2:
            score += 0.2
            reasons.append("Contains balance values")
            
        if self.cr_dr_re.search(row_text):
            score += 0.2
            reasons.append("Contains debit/credit indicators")
            
        if self.transaction_keywords.search(row_text):
            score += 0.3
            reasons.append("Contains transaction keywords")
            
        # Threshold for considering it a transaction candidate
        is_candidate = score >= 0.5
        
        return DetectionResult(
            is_transaction_candidate=is_candidate,
            confidence_score=min(1.0, score),
            reasons=reasons
        )
