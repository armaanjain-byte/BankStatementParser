"""
scratch/trace_stmt11.py  —  Full pipeline trace for statement 11.

Uses the failure_reports.json candidates + full row reconstruction from the 
evaluation report data to trace every row through the pipeline.
"""
from __future__ import annotations

import sys, re, json
sys.path.insert(0, ".")

from src.constants import DATE_RE, MONEY_RE
from src.date_validator import validate_date_candidate
from src.transaction_detector import TransactionCandidateDetector
from src.row_classifier import RowClassifier
from src.transaction_parser import HEADER_FOOTER_KEYWORDS

# ── Reconstruct statement 11 rows from the cached outputs OCR cache ──────────
# No OCR cache for 11, so we reconstruct using the failure_reports data instead.
# We use the full failure_reports which includes the reconstructed row text.

failures = json.load(open("out/failure_reports.json"))
s11_fails = [r for r in failures if str(r.get("statement_id", "")) == "11"]

print("=" * 100)
print("STATEMENT 11 — COMPLETE PIPELINE TRACE")
print("=" * 100)
print(f"\nTotal failure records for statement 11: {len(s11_fails)}")
print()

# ── Analyse the date formats present ─────────────────────────────────────────
print("=" * 100)
print("PART 1: DATE FORMAT ANALYSIS FROM FAILURE RECORDS")
print("=" * 100)

# Group by type
date_parse_failed = [r for r in s11_fails if r["failure_type"] == "date_parse_failed"]
no_date_detected  = [r for r in s11_fails if r["failure_type"] == "no_date_detected"]

print(f"\n  date_parse_failed : {len(date_parse_failed)}")
print(f"  no_date_detected  : {len(no_date_detected)}")

print("\n--- date_parse_failed candidates ---")
for r in date_parse_failed:
    print(f"  candidate: '{r['candidate']}'")

print("\n--- no_date_detected candidates ---")
for r in no_date_detected:
    print(f"  candidate: '{r['candidate']}'")

# ── Simulate what DATE_RE extracts from each candidate ───────────────────────
print()
print("=" * 100)
print("PART 2: DATE_RE EXTRACTION FROM no_date_detected ROWS")
print("=" * 100)
print()
print("  These are rows that the detector scored as transaction candidates,")
print("  but DATE_RE could not find ANY date string in them.")
print()

for r in no_date_detected:
    cand = r["candidate"]
    m = DATE_RE.search(cand)
    # Also try MM/DD pattern (no year)
    mm_dd = re.search(r'\b(\d{1,2}/\d{2})\b', cand)
    print(f"  TEXT    : {cand}")
    print(f"  DATE_RE : {'FOUND: ' + repr(m.group(0)) if m else 'NO MATCH'}")
    print(f"  MM/DD   : {'FOUND: ' + repr(mm_dd.group(0)) if mm_dd else 'not found'}")
    print()

# ── Root cause: MM/DD format without year ─────────────────────────────────────
print("=" * 100)
print("PART 3: ROOT CAUSE — MM/DD WITHOUT YEAR")
print("=" * 100)
print("""
  Statement 11 is a CITIBANK (US) statement.
  
  US bank statements use the date format: MM/DD  (e.g. "01/12", "02/12", "05/12")
  — two digits for month, two digits for day, with NO year in the transaction row.
  
  DATE_RE Pattern 1 requires:  DD/MM/YYYY  (two-part number then a 2-4 digit year)
  Pattern 1 regex: \\b\\d{1,2}[/\\-]\\d{1,2}[/\\-]\\d{2,4}
  
  "01/12" has only two parts separated by one slash — it is a valid DATE_RE
  Pattern 1 candidate ONLY if there is a year. Without the year it is NOT matched.
  
  → All 8 no_date_detected rows contain "MM/DD" dates without a year.
  → The 29 date_parse_failed rows are rows whose OCR-extracted "date" is actually
    a monetary amount like "7.87 64", "73.83 63", etc. — the balance column value
    was incorrectly selected as the date candidate by the reconstruction.
""")

# ── Verify: what does DATE_RE match in the date_parse_failed rows? ──────────
print("=" * 100)
print("PART 4: DATE_RE EXTRACTION FROM date_parse_failed ROWS")
print("=" * 100)
print()
print("  These are rows where DATE_RE DID match something, but validate_date_candidate")
print("  rejected it.")
print()

from src.date_validator import validate_date_candidate
for r in date_parse_failed:
    cand = r["candidate"]
    m = DATE_RE.search(cand)
    if m:
        is_valid, reason = validate_date_candidate(m.group(0))
        print(f"  TEXT    : {cand}")
        print(f"  MATCHED : '{m.group(0)}'")
        print(f"  VALID   : {is_valid}  REASON: {reason}")
        print()
    else:
        print(f"  TEXT    : {cand}")
        print(f"  MATCHED : (no match — row never reached date validation)")
        print()

# ── Identify all genuine transaction rows ────────────────────────────────────
print("=" * 100)
print("PART 5: GENUINE TRANSACTION ROWS IN STATEMENT 11")
print("=" * 100)
print("""
  Based on the candidate texts, the genuine transaction rows follow this pattern:
  
    MM/DD  <ref_code>  <DESCRIPTION>  <debit_or_credit>  <balance>
  
  Examples (from no_date_detected candidates):
""")
for r in no_date_detected:
    print(f"  {r['candidate']}")

print()
print("  Each row has a MM/DD date at the start (no year).")
print("  DATE_RE has no pattern that matches MM/DD without a year.")
print()

# ── Countable recoverable transactions ──────────────────────────────────────
print("=" * 100)
print("PART 6: RECOVERY ESTIMATE")
print("=" * 100)
print(f"""
  Recoverable transactions from no_date_detected rows : {len(no_date_detected)}
  
  Each of the {len(no_date_detected)} 'no_date_detected' rows is a genuine transaction
  with a MM/DD date at the start.

  The 29 'date_parse_failed' rows are NOT genuine transactions — they are either:
    a) Split-off balance/amount fragments from the same transaction rows
    b) Header/footer/running balance rows that contain amount-like numbers

  Adding a Pattern 9 to DATE_RE for bare MM/DD would recover all {len(no_date_detected)} transactions.
""")

# ── Propose the fix ──────────────────────────────────────────────────────────
print("=" * 100)
print("PART 7: PROPOSED FIX")
print("=" * 100)

TEST_DATES = ["01/12", "02/12", "05/12", "06/12", "15/12", "17/12", "08/12", "20/12"]
print("\n  New Pattern 9 candidate (MM/DD, no year, strict word boundary):")
PATTERN9 = re.compile(r'(?<!\d)(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])(?!\d|/\d)')
print(f"  regex: {PATTERN9.pattern}")
print()
for d in TEST_DATES:
    m = PATTERN9.search(d)
    print(f"    '{d}' → {'MATCH: ' + repr(m.group(0)) if m else 'NO MATCH'}")

# Also verify it does NOT match amounts
AMOUNTS = ["63,987.49", "64,039.79", "7.87", "73.83"]
print()
print("  Verify it does NOT match amounts:")
for a in AMOUNTS:
    m = PATTERN9.search(a)
    print(f"    '{a}' → {'MATCH (BAD!)' if m else 'no match (good)'}")

print()
print("  Also verify it does NOT match YYYY-MM-like year prefixes:")
FALSE_POS = ["2021-03/15", "01/2021"]
for f in FALSE_POS:
    m = PATTERN9.search(f)
    print(f"    '{f}' → {'MATCH (BAD!)' if m else 'no match (good)'}")
