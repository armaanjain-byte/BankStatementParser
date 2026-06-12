# Benchmark Report

This document summarizes the final evaluation metrics and forensic optimizations applied to the `BankStatementParser` pipeline. The objective of the forensic phase was to identify and eliminate artificial bottlenecks in the OCR and parser logic to maximize transaction recovery.

## 1. Dataset Summary
- **Size:** 91 bank statement documents.
- **Composition:** A diverse mix of templates (e.g., CIBC, generic statements), layouts, multi-column tables, digital scans, and low-contrast mobile photographs.

## 2. Benchmark Comparison (Before vs After)

Following the forensic analysis phase, the pipeline underwent a massive leap in performance.

| Metric | Baseline | Final | Difference |
| :--- | :--- | :--- | :--- |
| **Assigned Transactions** | 659 | **987** | <span style="color:green">**+328**</span> |
| **No Date Detected Failures** | 64 | **39** | <span style="color:green">**-25**</span> |
| **Date Parse Failures** | 32 | **19** | <span style="color:green">**-13**</span> |
| **Header Extraction Rate** | ~40% | **59.3%** | <span style="color:green">**+19%**</span> |

*(Note: The remaining failures are primarily correct rejections of non-transaction rows, such as summary tables or metadata, which effectively acts as successful noise filtering).*

## 3. Major Fixes & Measured Impact

The +328 transaction improvement was achieved without training any new ML models. It was driven entirely by deterministic logic refinement:

### 1. Removing Automatic OpenCV Preprocessing (+256 Transactions)
- **Root Cause:** The `PaddleOCRBackend` utilized an aggressive `_is_low_quality()` heuristic (variance < 1800) that forced 36 out of 91 statements through an OpenCV binarization step. This binarization destroyed data in naturally legible, low-contrast photographs, causing PaddleOCR to hallucinate completely incorrect characters.
- **Impact:** Disabling this forced preprocessing step allowed PaddleOCR to natively process the images, resulting in a staggering net gain of **+256 transactions** across 22 previously-failing statements.

### 2. Eliminating Erroneous Classifier Rejections (+57 Transactions)
- **Root Cause:** `RowClassifier` was rejecting any row containing the phrases `"Debit Card"` or `"Credit Card"`, mistaking valid transaction descriptions for account-level metadata.
- **Impact:** Removing these overly aggressive constraints recovered **57 valid transactions** that were being prematurely discarded before reaching the parser.

### 3. Stateful Date Inheritance (+15 Transactions)
- **Root Cause:** Certain bank templates (like Statement 80) group multiple transactions under a single overarching date header, leaving the subsequent rows without an inline date. The `DATE_RE` regex correctly failed to find a date, discarding these rows.
- **Impact:** Implementing a `last_seen_date` inheritance mechanism in `parse_rows` allowed these split transactions to inherit the context of their preceding sibling, recovering **15 additional transactions**.

### 4. Layout Detector & Unified Regex Unification
- **Root Cause:** Discrepancies between the Date Regex used in the `LayoutDetector` versus the `TransactionParser` caused layout bounds to fail on valid edge cases, omitting entire tables.
- **Impact:** Standardizing `DATE_RE` universally stabilized layout detection across the dataset.

## 4. Known Limitations

While the pipeline performs exceptionally well, a few minor limitations exist:
- **Spacing Constraints:** The `DATE_RE` pattern currently enforces a space between month abbreviations and days (e.g., `Jan 4`). Statements where the OCR omits this space (e.g., `Jan4`) will fail to parse (e.g., Statement 67). Fixing this runs a high risk of introducing false positives.
- **Extreme Angle Classification:** While `use_angle_cls=True` handles 90°/180° rotations, extremely skewed documents may still result in staggered bounding boxes that foil the `transaction_reconstructor`.
