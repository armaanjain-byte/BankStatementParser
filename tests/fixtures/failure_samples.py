"""
tests/fixtures/failure_samples.py — Verbatim samples from out/failure_reports.json.

These strings are copied directly from the failure report produced by
src/evaluation.py during the latest dataset run (statements 23-28; the
only statements with cached OCR output in this repo). They anchor the
regression suite to real observed failures rather than hypothetical ones.

NOTE ON STATEMENT IDS: the task brief references statements 11, 31, 37,
66, 70, 84, 86, 88, 90. Those statement IDs do not exist in this repo —
out/failure_reports.json and statement_outputs/ only cover IDs 23-28
(6 statements total). The regression fixtures below are built from the
failure records that actually exist for statements 23-27. If/when a
larger dataset (including statements 11/31/37/.../90) is added, extend
this file with their corresponding 'candidate' strings using the same
pattern.
"""

from __future__ import annotations

# ──────────────────────────────────────────────
#  no_date_detected failures (10 total in report)
# ──────────────────────────────────────────────
# These are the first 1-3 tokens of rows where DATE_RE found nothing
# AND the broader DATE_CANDIDATE_RE (evaluation.py) also found nothing.
NO_DATE_DETECTED_SAMPLES = [
    {"statement_id": "23", "candidate": "Sept. Bill paymnemt Irotn) '201053 JU [68233On-..."},
    {"statement_id": "23", "candidate": "(et . Withdrawal RarcWays;"},
    {"statement_id": "23", "candidate": "Oct . Cill parment from Garelars Four Oaks 32.0..."},
    {"statement_id": "23", "candidate": "End balance 302!41.139"},
    {"statement_id": "25", "candidate": "Jul 2017 ATM OPERATOR FEE WIThDRAWAL AT ANZ ATM"},
    {"statement_id": "25", "candidate": "AEN33 Do7 457 141 &AFSL NJshaHrt ccdi: |cajo 23..."},
    {"statement_id": "26", "candidate": "CLO-ZQ Z0E 17(M"},
    {"statement_id": "26", "candidate": "0R-UZ-20 NEFL-AXISI4A9004593-BASIREDDY Suuu.UU"},
    {"statement_id": "27", "candidate": "(0s-10-2021 BampvnicnI LOQU.UO"},
    {"statement_id": "27", "candidate": "04-2W2 MANPOKARUR VS 1676,0"},
]

# ──────────────────────────────────────────────
#  date_parse_failed failures (4 total in report)
# ──────────────────────────────────────────────
# These are DATE_CANDIDATE_RE matches that DATE_RE rejected outright
# (i.e. they look date-shaped to the loose candidate regex but not to
# the stricter production DATE_RE). All four are actually amount-shaped
# strings ("NN-N.NN") that were never real dates.
DATE_PARSE_FAILED_SAMPLES = [
    {"statement_id": "25", "candidate": "1/13.00"},
    {"statement_id": "26", "candidate": "00-6.00"},
    {"statement_id": "26", "candidate": "00-7.00"},
    {"statement_id": "26", "candidate": "00-7.00"},
]

# ──────────────────────────────────────────────
#  unhandled_exception failures (85 total in report)
# ──────────────────────────────────────────────
# Every statement (1, 2, 10-22, 29...) other than 23-28 failed with this
# exact exception because paddleocr is not installed in this environment.
# This is an environment/dependency issue, not a parsing-logic bug, but
# it must be guarded against so the e2e suite doesn't silently no-op.
UNHANDLED_EXCEPTION_SAMPLE = {
    "failure_type": "unhandled_exception",
    "candidate": "No module named 'paddleocr'",
}

# ──────────────────────────────────────────────
#  insufficient_numeric_values
# ──────────────────────────────────────────────
# pipeline_loss_report.json shows 0 occurrences of this category in the
# current run (count is 0 for statements 23-28). The regression test for
# this category therefore exercises the *logic path* in
# transaction_parser.parse_rows / evaluation.py using a synthetic row
# (a date with no amount) rather than a real captured sample, so the
# branch remains covered even though it produced zero hits historically.
INSUFFICIENT_NUMERIC_VALUES_SYNTHETIC = {
    "row_text": "12/03/2021 Interest credited for the quarter",
    "expected_failure_type": "insufficient_numeric_values",
}

# ──────────────────────────────────────────────
#  "unknown" failures (62 total — largest bucket in pipeline_loss_report)
# ──────────────────────────────────────────────
# pipeline_loss_report.json's ranked_failure_categories shows 62 "unknown"
# entries. Cross-referencing failure_reports.json, every individual record
# has a recognized failure_type (no_date_detected/date_parse_failed/
# unhandled_exception) -- there is NO record with failure_type=="unknown"
# in the per-record list. This strongly suggests "unknown" in the ranked
# summary is a DERIVED/AGGREGATE bucket computed elsewhere (e.g. rows that
# were dropped without ANY failure_reporter.report_failure call at all --
# see evaluation.py's `unknown_failure_count` branches, which increment a
# counter but do not always call report_failure).
#
# This is flagged as a genuine ambiguity for the test suite: the counts in
# pipeline_loss_report.json (62 unknown) cannot be fully reconciled against
# failure_reports.json (0 records tagged "unknown"). The regression test
# in tests/regression/test_known_failures.py documents this discrepancy
# and asserts the *current* (possibly buggy) behavior so any future fix
# to evaluation.py's counting/reporting consistency is a deliberate,
# visible change rather than a silent one.
UNKNOWN_FAILURE_COUNT_IN_LOSS_REPORT = 62
UNKNOWN_FAILURE_RECORDS_IN_FAILURE_REPORTS = 0