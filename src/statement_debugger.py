"""
statement_debugger.py — Forensic debugging for 0%-parse statements.

This module provides detailed tracing through the extraction pipeline for
statements that have 0% parse rate, identifying where rows are lost.

Trace output shows:
  OCR → Reconstruction → Classification → Normalization → Date Extraction → 
  Semantic Assignment

This enables identification of the first pipeline stage that loses rows,
which guides focused remediation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.constants import DATE_RE, MONEY_RE
from src.row_classifier import RowClassifier
from src.transaction_parser import parse_rows as parse_rows_default
from src.transaction_parser import DATE_RE as parser_date_re
from src.transaction_reconstructor import Row

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageMetrics:
    """Metrics for one pipeline stage."""
    stage_name: str
    rows_in: int
    rows_out: int
    rows_lost: int
    
    @property
    def loss_rate(self) -> float:
        """Percentage of rows lost at this stage."""
        return (self.rows_lost / self.rows_in * 100) if self.rows_in > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.stage_name}: {self.rows_out}/{self.rows_in} ({100-self.loss_rate:.0f}% kept)"


@dataclass
class PipelineDebugTrace:
    """Complete trace of a statement through the pipeline."""
    statement_id: str
    stages: List[PipelineStageMetrics]
    bottleneck_stage: Optional[str]
    
    def bottleneck(self) -> PipelineStageMetrics:
        """Find the stage with highest loss rate."""
        if not self.stages:
            return None
        max_loss = max(self.stages, key=lambda s: s.loss_rate)
        return max_loss if max_loss.loss_rate > 0 else None
    
    def summary(self) -> str:
        """Generate summary of pipeline trace."""
        lines = [f"Statement {self.statement_id}:"]
        for stage in self.stages:
            lines.append(f"  {stage}")
        
        bn = self.bottleneck()
        if bn:
            lines.append(f"  → BOTTLENECK: {bn.stage_name} ({bn.loss_rate:.0f}% loss)")
        else:
            lines.append(f"  → All rows passed through (no bottleneck detected)")
        
        return "\n".join(lines)


class StatementDebugger:
    """Trace a statement through the extraction pipeline."""
    
    def __init__(self, statement_id: str):
        self.statement_id = statement_id
        self.classifier = RowClassifier()
        self.stages: List[PipelineStageMetrics] = []
    
    def trace(self, rows: List[Row]) -> PipelineDebugTrace:
        """Trace rows through the pipeline.
        
        Args:
            rows: List of reconstructed Row objects from one statement
        
        Returns:
            PipelineDebugTrace with metrics for each stage
        """
        current_rows = rows
        self.stages = []
        
        # Stage 1: Input (reconstruction complete)
        self.stages.append(PipelineStageMetrics(
            stage_name="Input (Reconstruction)",
            rows_in=len(rows),
            rows_out=len(rows),
            rows_lost=0
        ))
        
        # Stage 2: Row classification (filter headers/footers)
        classified_rows = [
            r for r in current_rows
            if self.classifier.classify(r.text).is_transaction_candidate
        ]
        self.stages.append(PipelineStageMetrics(
            stage_name="Classification (Headers/Footers Filtered)",
            rows_in=len(current_rows),
            rows_out=len(classified_rows),
            rows_lost=len(current_rows) - len(classified_rows)
        ))
        current_rows = classified_rows
        
        # Stage 3: Date extraction
        rows_with_dates = [
            r for r in current_rows
            if DATE_RE.search(r.text)
        ]
        self.stages.append(PipelineStageMetrics(
            stage_name="Date Extraction (DATE_RE)",
            rows_in=len(current_rows),
            rows_out=len(rows_with_dates),
            rows_lost=len(current_rows) - len(rows_with_dates)
        ))
        current_rows = rows_with_dates
        
        # Stage 4: Amount extraction
        rows_with_amounts = [
            r for r in current_rows
            if MONEY_RE.search(r.text)
        ]
        self.stages.append(PipelineStageMetrics(
            stage_name="Amount Extraction (MONEY_RE)",
            rows_in=len(current_rows),
            rows_out=len(rows_with_amounts),
            rows_lost=len(current_rows) - len(rows_with_amounts)
        ))
        current_rows = rows_with_amounts
        
        # Stage 5: Parsing (transaction_parser)
        parsed_rows = parse_rows_default(Row.copy_list(current_rows))
        self.stages.append(PipelineStageMetrics(
            stage_name="Transaction Parsing",
            rows_in=len(current_rows),
            rows_out=len(parsed_rows),
            rows_lost=len(current_rows) - len(parsed_rows)
        ))
        
        return PipelineDebugTrace(
            statement_id=self.statement_id,
            stages=self.stages,
            bottleneck_stage=self._find_bottleneck()
        )
    
    def _find_bottleneck(self) -> Optional[str]:
        """Identify which stage loses the most rows."""
        if not self.stages or all(s.loss_rate == 0 for s in self.stages):
            return None
        
        worst = max(self.stages, key=lambda s: s.loss_rate)
        return worst.stage_name if worst.loss_rate > 0 else None
    
    def analyze_lost_rows(self, rows: List[Row], failed_indices: set[int]) -> dict:
        """Analyze characteristics of rows that failed.
        
        Args:
            rows: All rows
            failed_indices: Set of row indices that failed
        
        Returns:
            Dict with failure patterns and suggestions
        """
        failed_rows = [rows[i] for i in failed_indices if i < len(rows)]
        
        analysis = {
            "total_failed": len(failed_rows),
            "patterns": {
                "no_date": 0,
                "no_amount": 0,
                "header_like": 0,
                "footer_like": 0,
                "page_marker": 0,
                "short_text": 0,
                "separator": 0,
            },
            "examples": []
        }
        
        for row in failed_rows[:50]:  # Analyze first 50
            text = row.text
            
            # Categorize failure
            if not DATE_RE.search(text):
                analysis["patterns"]["no_date"] += 1
            
            if not MONEY_RE.search(text):
                analysis["patterns"]["no_amount"] += 1
            
            classification = self.classifier.classify(text)
            if classification.classification == "header":
                analysis["patterns"]["header_like"] += 1
            elif classification.classification == "footer":
                analysis["patterns"]["footer_like"] += 1
            elif classification.classification == "page_marker":
                analysis["patterns"]["page_marker"] += 1
            
            if len(text) < 10:
                analysis["patterns"]["short_text"] += 1
            
            if all(c in " -_=" for c in text):
                analysis["patterns"]["separator"] += 1
            
            # Store examples
            if len(analysis["examples"]) < 5:
                analysis["examples"].append(text[:80])
        
        return analysis


class Row:
    """Helper to copy Row objects without OCRToken references."""
    
    @staticmethod
    def copy_list(rows: List) -> List:
        """Create lightweight copy of rows for tracing."""
        from src.transaction_reconstructor import Row as OrigRow
        
        copies = []
        for row in rows:
            copy = OrigRow(
                y_center=row.y_center,
                tokens=[],
                text=row.text,
                original_text=row.original_text
            )
            copies.append(copy)
        
        return copies


def debug_statement(statement_id: str, rows: List[Row]) -> str:
    """Generate complete debug report for a statement.
    
    Args:
        statement_id: The statement identifier
        rows: Reconstructed rows from the statement
    
    Returns:
        Formatted debug report as string
    """
    debugger = StatementDebugger(statement_id)
    trace = debugger.trace(rows)
    
    report = [
        "=" * 70,
        f"STATEMENT DEBUG TRACE: {statement_id}",
        "=" * 70,
        "",
        trace.summary(),
        ""
    ]
    
    # Add recommendation
    bottleneck = trace.bottleneck()
    if bottleneck:
        report.append(f"FOCUS AREA: {bottleneck.stage_name}")
        if "Classification" in bottleneck.stage_name:
            report.append("  → Review row_classifier patterns for false positives")
        elif "Date Extraction" in bottleneck.stage_name:
            report.append("  → Check for missing date patterns or amount false positives")
        elif "Amount Extraction" in bottleneck.stage_name:
            report.append("  → Check MONEY_RE pattern or normalization issues")
        elif "Parsing" in bottleneck.stage_name:
            report.append("  → Check semantic assignment logic or layout detection")
    
    report.append("=" * 70)
    
    return "\n".join(report)