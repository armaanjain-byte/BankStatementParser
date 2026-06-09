"""
analyze_dataset.py — Exploratory analysis of OCR output across the dataset.

Reads all ``*_ocr.json`` files from the outputs directory and produces a
``dataset_report.json`` summarizing:

  • Token counts and confidence statistics per file
  • Most common potential bank names detected
  • Currency symbols / codes found in OCR text
  • Date-format patterns observed

This is a **read-only** tool — it never modifies OCR outputs.

Usage:
    python -m src.analyze_dataset                       # defaults
    python -m src.analyze_dataset -i outputs -o report  # custom dirs
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.constants import CURRENCY_MAP, SYMBOL_TO_CURRENCY, SUPPORTED_BANKS

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Detection patterns
# ──────────────────────────────────────────────

# Date-format regexes, ordered from most specific to least.
# Each entry is (label, compiled regex).
DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # DD/MM/YYYY or DD-MM-YYYY
    ("DD/MM/YYYY",  re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b")),
    # DD/MM/YY or DD-MM-YY
    ("DD/MM/YY",    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2}\b")),
    # YYYY-MM-DD (ISO)
    ("YYYY-MM-DD",  re.compile(r"\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b")),
    # DD Mon YYYY  (e.g. 01 Apr 2025, 1 Jan 2025)
    ("DD Mon YYYY", re.compile(
        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    )),
    # DD Mon YY  (e.g. 01 Apr 25)
    ("DD Mon YY", re.compile(
        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{2}\b",
        re.IGNORECASE,
    )),
    # Mon DD, YYYY  (e.g. Apr 01, 2025)
    ("Mon DD, YYYY", re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    )),
    # DD Mon  (e.g. "19 Sept" — no year, common in multi-page statements)
    ("DD Mon", re.compile(
        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.]*\b",
        re.IGNORECASE,
    )),
]

# Currency symbols to search for in raw text.
# Built from constants.py + a few common OCR-mangled variants.
_CURRENCY_SYMBOLS: dict[str, str] = {
    **SYMBOL_TO_CURRENCY,
    "Rs": "INR",
    "Rs.": "INR",
    "INR": "INR",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
}

# Bank name fragments to search for (case-insensitive substrings).
# We include SUPPORTED_BANKS plus common short forms found on statements.
_BANK_KEYWORDS: dict[str, str] = {}
for bank in SUPPORTED_BANKS:
    # Map full name → full name
    _BANK_KEYWORDS[bank.lower()] = bank
# Also add common short-forms / abbreviations
_BANK_KEYWORDS.update({
    "sbi":           "State Bank of India",
    "hdfc":          "HDFC Bank",
    "icici":         "ICICI Bank",
    "axis":          "Axis Bank",
    "pnb":           "Punjab National Bank",
    "bob":           "Bank of Baroda",
    "canara":        "Canara Bank",
    "kotak":         "Kotak Mahindra Bank",
    "indusind":      "IndusInd Bank",
    # Add non-SUPPORTED banks we might discover in the dataset
    "hsbc":          "HSBC",
    "barclays":      "Barclays",
    "standard chartered": "Standard Chartered",
    "citi":          "Citibank",
    "citibank":      "Citibank",
    "rbs":           "Royal Bank of Scotland",
    "lloyds":        "Lloyds Bank",
    "natwest":       "NatWest",
    "deutsche":      "Deutsche Bank",
    "ubs":           "UBS",
    "jpmorgan":      "JPMorgan Chase",
    "chase":         "JPMorgan Chase",
    "wells fargo":   "Wells Fargo",
    "bank of america": "Bank of America",
    "scotia":        "Scotiabank",
    "td bank":       "TD Bank",
    "yes bank":      "Yes Bank",
    "idbi":          "IDBI Bank",
    "federal bank":  "Federal Bank",
    "bandhan":       "Bandhan Bank",
})


# ──────────────────────────────────────────────
#  Per-file analysis
# ──────────────────────────────────────────────

def _analyze_file(data: dict) -> dict[str, Any]:
    """Analyze a single OCR JSON payload and return per-file stats."""
    tokens = data.get("tokens", [])
    texts = [t["text"] for t in tokens]
    confidences = [t["confidence"] for t in tokens]

    # ── Confidence stats ──────────────────
    n = len(confidences)
    avg_conf = sum(confidences) / n if n else 0.0
    min_conf = min(confidences) if n else 0.0
    max_conf = max(confidences) if n else 0.0
    low_conf_count = sum(1 for c in confidences if c < 0.40)
    high_conf_count = sum(1 for c in confidences if c >= 0.85)

    # ── Full text blob for pattern matching ──
    full_text = " ".join(texts)
    full_lower = full_text.lower()

    # ── Bank detection ────────────────────
    banks_found: list[str] = []
    for keyword, canonical in _BANK_KEYWORDS.items():
        if keyword in full_lower:
            banks_found.append(canonical)

    # ── Currency detection ────────────────
    currencies_found: list[str] = []
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in full_text:
            currencies_found.append(code)
    # deduplicate while preserving order
    currencies_found = list(dict.fromkeys(currencies_found))

    # ── Date format detection ─────────────
    date_formats: dict[str, list[str]] = {}
    for label, pattern in DATE_PATTERNS:
        matches = pattern.findall(full_text)
        if matches:
            # Keep up to 5 example matches per format
            date_formats[label] = matches[:5]

    return {
        "token_count": n,
        "confidence": {
            "average": round(avg_conf, 4),
            "min": round(min_conf, 4),
            "max": round(max_conf, 4),
            "low_confidence_tokens": low_conf_count,
            "high_confidence_tokens": high_conf_count,
        },
        "banks_detected": sorted(set(banks_found)),
        "currencies_detected": currencies_found,
        "date_formats": date_formats,
    }


# ──────────────────────────────────────────────
#  Aggregate statistics
# ──────────────────────────────────────────────

def _aggregate(per_file: dict[str, dict]) -> dict[str, Any]:
    """Compute dataset-wide aggregate statistics from per-file results."""
    total_files = len(per_file)
    if total_files == 0:
        return {}

    all_token_counts = [f["token_count"] for f in per_file.values()]
    all_avg_confs = [f["confidence"]["average"] for f in per_file.values()]

    # Count bank name occurrences across all files
    bank_counter: Counter = Counter()
    for f in per_file.values():
        bank_counter.update(f["banks_detected"])

    # Count currency occurrences across all files
    currency_counter: Counter = Counter()
    for f in per_file.values():
        currency_counter.update(f["currencies_detected"])

    # Count date-format occurrences (how many files contain each format)
    date_format_counter: Counter = Counter()
    for f in per_file.values():
        for fmt in f["date_formats"]:
            date_format_counter[fmt] += 1

    return {
        "total_files": total_files,
        "tokens": {
            "total": sum(all_token_counts),
            "average_per_file": round(sum(all_token_counts) / total_files, 1),
            "min_per_file": min(all_token_counts),
            "max_per_file": max(all_token_counts),
        },
        "confidence": {
            "dataset_average": round(
                sum(all_avg_confs) / total_files, 4
            ),
            "best_file": max(per_file, key=lambda k: per_file[k]["confidence"]["average"]),
            "worst_file": min(per_file, key=lambda k: per_file[k]["confidence"]["average"]),
        },
        "bank_name_frequency": dict(bank_counter.most_common()),
        "currency_frequency": dict(currency_counter.most_common()),
        "date_format_frequency": dict(date_format_counter.most_common()),
    }


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def generate_report(
    input_dir: Path,
    output_path: Path,
) -> Path:
    """Scan all OCR JSON files and generate a dataset report.

    Args:
        input_dir:   Directory containing ``*_ocr.json`` files.
        output_path: Where to write ``dataset_report.json``.

    Returns:
        Path to the written report file.
    """
    json_files = sorted(input_dir.glob("*_ocr.json"))
    if not json_files:
        logger.error("No *_ocr.json files found in %s", input_dir)
        sys.exit(1)

    logger.info("Analyzing %d OCR file(s) in %s", len(json_files), input_dir)

    per_file: dict[str, dict] = {}

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats = _analyze_file(data)
        per_file[jf.name] = stats
        logger.info(
            "  %-25s | tokens: %4d | avg conf: %.4f | banks: %s",
            jf.name,
            stats["token_count"],
            stats["confidence"]["average"],
            ", ".join(stats["banks_detected"]) or "-",
        )

    aggregate = _aggregate(per_file)

    report = {
        "summary": aggregate,
        "per_file": per_file,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("-" * 60)
    logger.info("Report saved -> %s", output_path)
    logger.info(
        "Dataset: %d files, %d total tokens, avg confidence %.4f",
        aggregate["total_files"],
        aggregate["tokens"]["total"],
        aggregate["confidence"]["dataset_average"],
    )
    logger.info(
        "Banks found: %s",
        ", ".join(aggregate["bank_name_frequency"].keys()) or "none",
    )
    logger.info(
        "Currencies found: %s",
        ", ".join(aggregate["currency_frequency"].keys()) or "none",
    )
    logger.info(
        "Date formats found: %s",
        ", ".join(aggregate["date_format_frequency"].keys()) or "none",
    )

    return output_path


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze OCR JSON outputs and generate a dataset report.",
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing *_ocr.json files (default: outputs/)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("dataset_report.json"),
        help="Output report path (default: dataset_report.json)",
    )
    args = parser.parse_args()

    generate_report(args.input_dir, args.output)
