# Known Limitations

While `BankStatementParser` is highly performant and specifically optimized for a wide variety of statement templates, it is a rules-based system. As such, there are certain edge cases and systemic limitations to be aware of.

## 1. Strict Date Regex Constraints
The parsing layer relies heavily on the `DATE_RE` regular expression pattern to detect valid transactions. 
- **Missing Spacing:** Currently, the pattern mandates a space between month abbreviations and single/double-digit days if no year is provided (e.g., `Jan 4` is valid, but `Jan4` is invalid). If the OCR backend swallows this space, the row will fail to parse (e.g., Statement 67). Modifying this pattern to allow optional spacing (`\s*`) runs a high risk of false-positive matches on random alphanumeric descriptions.
- **Date Pre-fixing:** Dates attached directly to prefix strings without delimiters (e.g., `S9703109702/04/2017` instead of `02/04/2017`) require specific regex branches. While we implemented a fix for Statement 3 (`TargetAttachedPrefixDate`), previously unseen prefix formats will still fail.

## 2. Dependency on High-Fidelity OCR Layouts
The Row Reconstructor groups tokens based on overlapping Y-axis coordinates. 
- **Extreme Document Skew:** If a document is scanned at an extreme angle that surpasses `PaddleOCR`'s rotation correction capabilities, the text tokens will stair-step. This causes tokens belonging to the same logical row to fall into different horizontal buckets, fracturing the transaction into unusable fragments.
- **Micro-columns:** If transaction amounts and dates are placed extremely close to each other without sufficient whitespace, the OCR may merge them into a single string (e.g., `12.27.00`), causing amount parsing failures.

## 3. Disabling of OpenCV Preprocessing
To maximize dataset recovery, the automatic OpenCV preprocessing heuristic in `PaddleOCRBackend` was permanently disabled.
- **Trade-off:** OpenCV binarization was destroying valid data on lower-contrast digital photographs (costing us over 250 transactions). However, there were 3 known statements in our benchmark that experienced extremely minor regressions (-1 to -3 transactions) due to the removal of this feature. 
- **Future Work:** If maximum absolute recovery is required, a Dual-Pass OCR architecture should be implemented. This would run the raw image first, and if parsing fails, fallback to an OpenCV preprocessed version of the image.

## 4. Lack of Multi-Page Table Stitching
The system treats every image independently. If a single transaction description spans across a page break (the end of one page and the start of the next), it will be truncated or fail to parse. It currently does not stitch tables temporally across page boundaries.
