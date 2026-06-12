# BankStatementParser 🏦

A fully deterministic, high-performance pipeline for extracting transaction data from bank statement images and PDFs.

## Project Overview
BankStatementParser is designed to process raw, unstructured bank statement scans and photos, transforming them into structured semantic data (JSON). Rather than relying on black-box Large Language Models (LLMs) which are slow, expensive, and prone to hallucinations, this pipeline utilizes a rigorous, rules-based architecture on top of PaddleOCR. This ensures deterministic parsing, lightning-fast execution speeds, and complete forensic auditability of every extracted row.

## Architecture at a Glance
The system is divided into modular, highly-testable components:
1. **OCR Backend:** Uses PaddleOCR to extract text tokens and precise bounding boxes.
2. **Row Reconstructor:** Groups tokens into logical visual rows using spatial Y-axis proximity.
3. **Normalizer:** Standardizes monetary values and date formats across various bank templates.
4. **Layout Detector:** Identifies the structural bounds of the statement (headers, footers, transaction tables).
5. **Transaction Parser:** Extracts dates, descriptions, and amounts using highly-optimized regex patterns.
6. **Semantic Assigner:** Validates the parsed rows against the detected layout, producing the final structured JSON.

For a deep dive into the system design, see the [Architecture Documentation](docs/Architecture.md).

## Benchmark Performance
Following extensive forensic optimizations, the parser achieves a staggering recovery rate on real-world datasets:
- **Total Transactions Assigned:** **987** (a +328 improvement over baseline)
- **Header Extraction Success Rate:** 59.34%
- **Average Validation Confidence:** 77.03%
- **Average Layout Confidence:** 77.86%

For full details, view the [Benchmark Report](docs/Benchmark_Report.md).

## Installation & Setup

### Requirements
- Python 3.9+
- GPU recommended (for accelerated PaddleOCR inference)

### 1. Create a Virtual Environment
```bash
python -m venv paddle_env
# Windows:
paddle_env\Scripts\activate
# Linux/Mac:
source paddle_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: PaddleOCR will automatically download its pre-trained inference models upon first execution).*

## Usage Guide

### Processing a Single Statement
To process a single bank statement image and output the structured JSON:
```bash
python evaluate_dataset.py <path_to_image>
```

### Running the Full Benchmark Suite
To run the evaluation across a complete dataset of images:
```bash
python evaluate_dataset.py dataset/ --clear-cache
```
Results will be saved in the `out/` directory, including:
- `statement_outputs/*.json`: The final structured extraction payload for each statement. View the [JSON Schema Documentation](docs/Schema.md) for data structure details.
- `evaluation_report.json`: Overall benchmark metrics
- `pipeline_loss_report.json`: Breakdown of assigned transactions per statement
- `failure_reports.json`: Detailed forensics of dropped or unparsed rows

## Known Limitations
For a comprehensive list of known edge cases (such as extreme OCR rotation limits and strict regex spacing requirements), please refer to the [Known Limitations](docs/Known_Limitations.md) guide.

## Testing
The pipeline is backed by a robust suite of over 300+ unit and integration tests.
```bash
pytest tests/
```
