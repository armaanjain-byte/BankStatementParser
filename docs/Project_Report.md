# Project Report: BankStatementParser
## Image Processing Techniques for Deterministic Bank Statement Transaction Extraction

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Image Processing Pipeline — Detailed Analysis](#5-image-processing-pipeline--detailed-analysis)
   - 5.1 [Image Acquisition and Input Handling](#51-image-acquisition-and-input-handling)
   - 5.2 [Image Preprocessing (OpenCV Pipeline)](#52-image-preprocessing-opencv-pipeline)
   - 5.3 [OCR Engine — PaddleOCR Deep Learning Backend](#53-ocr-engine--paddleocr-deep-learning-backend)
   - 5.4 [Bounding-Box Geometry and Spatial Reconstruction](#54-bounding-box-geometry-and-spatial-reconstruction)
   - 5.5 [OCR Error Correction and Normalization](#55-ocr-error-correction-and-normalization)
   - 5.6 [Layout Detection via Spatial Feature Analysis](#56-layout-detection-via-spatial-feature-analysis)
   - 5.7 [Semantic Assignment via Geometric Distance Matching](#57-semantic-assignment-via-geometric-distance-matching)
6. [Image Processing Techniques — Technical Deep Dive](#6-image-processing-techniques--technical-deep-dive)
   - 6.1 [Grayscale Conversion](#61-grayscale-conversion)
   - 6.2 [Deskewing via Minimum Area Rectangle](#62-deskewing-via-minimum-area-rectangle)
   - 6.3 [Non-Local Means Denoising](#63-non-local-means-denoising)
   - 6.4 [Adaptive Gaussian Thresholding (Binarization)](#64-adaptive-gaussian-thresholding-binarization)
   - 6.5 [Morphological Opening (Noise Cleanup)](#65-morphological-opening-noise-cleanup)
   - 6.6 [Canny Edge Detection and Hough Line Transform](#66-canny-edge-detection-and-hough-line-transform)
   - 6.7 [Affine Warp Transformation](#67-affine-warp-transformation)
   - 6.8 [PDF Rasterization](#68-pdf-rasterization)
   - 6.9 [PaddleOCR's Internal Image Processing](#69-paddleocrs-internal-image-processing)
7. [Post-OCR Text Processing Pipeline](#7-post-ocr-text-processing-pipeline)
   - 7.1 [Row Reconstruction Algorithm](#71-row-reconstruction-algorithm)
   - 7.2 [Date Normalization Pipeline](#72-date-normalization-pipeline)
   - 7.3 [Amount Normalization Pipeline](#73-amount-normalization-pipeline)
   - 7.4 [Row Classification and Transaction Detection](#74-row-classification-and-transaction-detection)
   - 7.5 [Transaction Parsing with Stateful Date Inheritance](#75-transaction-parsing-with-stateful-date-inheritance)
8. [Quality Assurance and Image-Driven Validation](#8-quality-assurance-and-image-driven-validation)
   - 8.1 [Low-Quality Image Detection Heuristic](#81-low-quality-image-detection-heuristic)
   - 8.2 [Confidence Scoring Framework](#82-confidence-scoring-framework)
   - 8.3 [Forensic Failure Analysis](#83-forensic-failure-analysis)
9. [Benchmark Results and Forensic Optimizations](#9-benchmark-results-and-forensic-optimizations)
10. [Known Limitations](#10-known-limitations)
11. [Technology Stack](#11-technology-stack)
12. [Testing Infrastructure](#12-testing-infrastructure)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Abstract

This report presents a comprehensive analysis of the **BankStatementParser** — a fully deterministic, rules-based pipeline for extracting structured transaction data from bank statement images and PDFs. The system transforms raw, unstructured scans and mobile photographs of bank statements into machine-readable JSON output, without relying on Large Language Models (LLMs). At the core of this system is a multi-stage image processing pipeline that leverages PaddleOCR for deep-learning-based text detection and recognition, OpenCV for classical image enhancement, and custom spatial-geometric algorithms for document layout understanding. The pipeline achieves a recovery rate of **987 assigned transactions** across a benchmark dataset of 91 diverse bank statement documents, with an average validation confidence of **77.03%**. This report provides an exhaustive technical analysis of every image processing technique employed, from pixel-level preprocessing through to geometric semantic assignment.

---

## 2. Introduction

Bank statements remain one of the most ubiquitous financial documents globally. Despite the proliferation of digital banking, a vast number of financial workflows — loan applications, audits, forensic accounting, regulatory compliance — still require processing physical or scanned bank statements. The challenge lies in converting these visually complex, heterogeneous documents into structured, machine-readable data.

Traditional approaches to this problem fall into two categories:

1. **LLM-based approaches** — Using large language models (GPT-4, Gemini, etc.) to interpret document images. While flexible, these are expensive per inference, non-deterministic (different runs can yield different outputs), prone to hallucination, and difficult to audit forensically.

2. **Rules-based approaches** — Using a combination of OCR, geometric analysis, and pattern matching. These are fast, deterministic, fully auditable, and cost-effective at scale.

BankStatementParser adopts the second approach, implementing a rigorous, modular pipeline where each stage transforms the data with complete traceability. The system is designed to process statements from diverse banks (HDFC, ICICI, SBI, Axis, PNB, international banks, etc.) across a wide range of input quality — from high-resolution digital scans to low-contrast mobile photographs.

---

## 3. Problem Statement

Extract structured transaction data (dates, descriptions, debit amounts, credit amounts, and running balances) from bank statement images and PDFs, where:

- Input images vary drastically in quality (digital scans, mobile photos, faxed copies).
- Document layouts differ across banks and even across statement periods from the same bank.
- OCR engines introduce character-level errors (e.g., `O` ↔ `0`, `l` ↔ `1`, `S` ↔ `$`).
- Table structures may lack explicit borders, use variable column widths, or span multiple pages.
- Transaction rows may be split across lines, share date headers, or contain merged cells.

The system must produce **deterministic, auditable** output with per-field confidence scores and detailed failure diagnostics.

---

## 4. System Architecture Overview

The BankStatementParser is designed as a **linear, single-pass pipeline** that progressively refines raw pixels into structured semantic objects. The architecture consists of seven core stages:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│   Image (JPG/PNG/TIFF/BMP/WebP) or PDF Document                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: IMAGE ACQUISITION & OPTIONAL PREPROCESSING                 │
│  • PDF rasterization (PyMuPDF, 300 DPI grayscale)                    │
│  • Optional OpenCV preprocessing (deskew, denoise, binarize)         │
│  • Low-quality detection heuristic (pixel variance analysis)         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: OCR TEXT EXTRACTION (PaddleOCR)                            │
│  • Deep learning text detection (DB algorithm)                       │
│  • Text recognition (CRNN + CTC/Attention decoder)                   │
│  • Angle classification for rotated documents                        │
│  • Output: List of OCRToken(text, confidence, bbox)                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: SPATIAL ROW RECONSTRUCTION                                 │
│  • Y-axis center computation for each token                          │
│  • Adaptive threshold clustering (median_height × 0.5)               │
│  • Left-to-right token sorting within clusters                       │
│  • Output: List of Row(y_center, tokens, text)                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: TEXT NORMALIZATION                                         │
│  • Date OCR error correction (3-pass: typos, interpolation, canon.)  │
│  • Amount OCR error correction (character substitution, separators)  │
│  • Unicode normalization (NFC, typographic character replacement)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: LAYOUT DETECTION                                           │
│  • Header row identification (alias matching with fuzzy fallback)    │
│  • Column role assignment (date, description, debit, credit, etc.)  │
│  • Data-row geometric inference fallback                             │
│  • Output: LayoutDefinition with column positions and confidence     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6: TRANSACTION PARSING & SEMANTIC ASSIGNMENT                  │
│  • Row classification (header/footer/metadata/transaction filtering) │
│  • Transaction candidate detection (signal scoring)                  │
│  • Date extraction with validation (9-pattern DATE_RE)               │
│  • Monetary value extraction (MONEY_RE)                              │
│  • Geometric distance-based column-value mapping                     │
│  • Stateful date inheritance for split transactions                  │
│  • Contextual year inference                                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 7: VALIDATION & OUTPUT                                        │
│  • Transaction completeness checks                                   │
│  • Layout consistency validation                                     │
│  • Confidence score computation                                      │
│  • Structured JSON output (BankStatement schema)                     │
│  • Failure reports and pipeline loss telemetry                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

| Module | Responsibility | Key Image Processing Technique |
|:---|:---|:---|
| `preprocessing.py` | OpenCV image enhancement | Grayscale, deskew, denoise, binarize, morphology |
| `backends/paddleocr_backend.py` | Deep learning OCR | DB text detection, CRNN recognition, angle classification |
| `ocr.py` | OCR orchestration | Confidence filtering, bbox normalization |
| `pdf_converter.py` | PDF → image rendering | Rasterization at 300 DPI |
| `transaction_reconstructor.py` | Spatial row clustering | Y-axis geometric clustering |
| `layout_detector.py` | Table structure analysis | X-axis column position analysis |
| `semantic_assigner.py` | Value-to-column mapping | Geometric distance matching |
| `backends/base.py` | Quality detection | Pixel variance analysis |

---

## 5. Image Processing Pipeline — Detailed Analysis

### 5.1 Image Acquisition and Input Handling

The pipeline accepts multiple input formats:

- **Raster images:** JPEG, PNG, TIFF, BMP, WebP
- **PDF documents:** Rendered to grayscale NumPy arrays at 300 DPI

For PDF inputs, the system uses **PyMuPDF (fitz)** to rasterize each page independently:

```python
def pdf_to_images(pdf_path, dpi=300):
    scale = dpi / 72          # PDF standard is 72 points/inch
    matrix = fitz.Matrix(scale, scale)
    for page in document:
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width
        )
```

**Key design decisions:**
- **300 DPI** was chosen as the optimal balance between OCR accuracy and memory consumption. At 300 DPI, standard A4 documents render to approximately 2480 × 3508 pixels, providing sufficient detail for PaddleOCR's text detection network.
- **Grayscale colorspace** (`fitz.csGRAY`) is used because bank statements are predominantly black text on white background; color information adds no value and triples memory usage.
- Each page is processed independently — there is currently no cross-page table stitching.

### 5.2 Image Preprocessing (OpenCV Pipeline)

The preprocessing module (`preprocessing.py`) implements a classical computer vision pipeline designed to enhance OCR accuracy on degraded inputs. It consists of five sequential stages:

#### Pipeline Flow:
```
Raw Image → Grayscale → Deskew → Denoise → Binarize → Morphological Cleanup → OCR-Ready Image
```

Each stage is implemented as an independent, composable function:

1. **Grayscale Loading:** `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` — converts the image to a single-channel 8-bit representation.

2. **Deskewing** (`_deskew`): Detects and corrects rotational skew up to ±15° using minimum area rectangle fitting on dark pixel coordinates.

3. **Denoising** (`_denoise`): Applies Non-Local Means Denoising to suppress Gaussian and salt-and-pepper noise while preserving text edges.

4. **Binarization** (`_binarize`): Converts the grayscale image to binary (black/white) using adaptive Gaussian thresholding, handling uneven illumination.

5. **Morphological Cleanup** (`_morphological_cleanup`): Removes isolated noise pixels using morphological opening with a 2×2 rectangular structuring element.

> **Critical Finding — Preprocessing Disabled by Default:**
> Benchmark analysis revealed that the automatic preprocessing pipeline was **destructive** on 36 out of 91 statements. The `_is_low_quality()` heuristic (pixel variance < 1800) incorrectly triggered binarization on naturally legible low-contrast photographs, causing PaddleOCR to hallucinate incorrect characters. **Disabling automatic preprocessing recovered +256 transactions** — the single largest improvement in the entire optimization cycle. The preprocessing module is retained for optional use (`preprocess=True`) but is no longer invoked by default.

### 5.3 OCR Engine — PaddleOCR Deep Learning Backend

PaddleOCR is the core vision layer of the pipeline. It is a deep-learning-based OCR system that performs three distinct image processing tasks internally:

#### 5.3.1 Text Detection (DB Algorithm)
PaddleOCR uses the **Differentiable Binarization (DB)** algorithm for text detection. DB is a segmentation-based text detector that:
- Takes the full image as input to a ResNet-based backbone network.
- Produces a probability map where each pixel indicates the likelihood of belonging to a text region.
- Applies a differentiable binarization step to convert the probability map to a binary mask.
- Extracts connected components as text region polygons (four-corner bounding boxes).

This approach is particularly effective for bank statements because it handles:
- Dense tabular layouts with closely-spaced text.
- Variable font sizes (headers vs. transaction data).
- Low-contrast text on colored or degraded backgrounds.

#### 5.3.2 Text Recognition (CRNN + CTC/Attention)
Each detected text region is cropped, resized, and passed through a **Convolutional Recurrent Neural Network (CRNN)**:
- **CNN feature extractor:** Extracts visual features from the cropped text image.
- **Bidirectional LSTM:** Models sequential dependencies between characters.
- **CTC (Connectionist Temporal Classification) decoder:** Produces the final character sequence.

The recognition network outputs both the recognized text string and a confidence score in [0.0, 1.0].

#### 5.3.3 Angle Classification
The pipeline initializes PaddleOCR with `use_angle_cls=True`, enabling a lightweight classification network that detects and corrects 0°/180° text orientation. This is critical for:
- Scanned documents fed through a scanner upside-down.
- Rotated mobile photographs.
- Mixed-orientation content (e.g., watermarks at 90°).

#### 5.3.4 Backend Configuration
```python
_ocr = PaddleOCR(
    lang="en",
    device="gpu",           # GPU acceleration for batch processing
    enable_mkldnn=False,    # Disabled for stability
    use_angle_cls=True      # Critical for rotated scans
)
result = ocr.ocr(img_input, cls=True)  # cls=True enables angle classification
```

#### 5.3.5 Token Extraction and Normalization
The raw PaddleOCR output is normalized into a canonical `OCRToken` format:

```python
@dataclass
class OCRToken:
    text: str                            # Recognized text
    confidence: float                    # [0.0, 1.0]
    bbox: Tuple[int, int, int, int]      # (x_min, y_min, x_max, y_max)
```

The backend handles two PaddleOCR API formats:
- **New dict-based API (≥ 2.8):** `rec_texts`, `rec_scores`, `rec_boxes` dictionaries.
- **Legacy list-based API:** `[[bbox_points], (text, confidence)]` tuples.

Polygon bounding boxes (four corner points) are converted to axis-aligned bounding boxes:
```python
bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
```

A minimum confidence threshold of **0.10** is applied to filter out noise tokens.

### 5.4 Bounding-Box Geometry and Spatial Reconstruction

The Row Reconstructor (`transaction_reconstructor.py`) is the critical bridge between raw OCR output and structured text. It operates purely on geometric coordinates — no text understanding is involved at this stage.

#### Algorithm:

1. **Compute vertical centers:** For each token, calculate `y_center = (y_min + y_max) / 2`.

2. **Sort tokens top-to-bottom** by `y_center`.

3. **Adaptive threshold computation:**
   ```python
   heights = [token.bbox[3] - token.bbox[1] for token in tokens]
   median_height = statistics.median(heights)
   threshold = median_height * 0.5  # Half the median character height
   ```
   This adaptive approach handles documents of any resolution — high-DPI scans with large pixel coordinates and low-DPI captures with small coordinates both get appropriate thresholds.

4. **Greedy Y-axis clustering:** Walk through sorted tokens, accumulating tokens into the current row while their `y_center` differs by ≤ `threshold` pixels from the group's running mean anchor:
   ```python
   if abs(y - anchor_y) <= threshold:
       current_group.append(token)
       anchor_y = running_mean_of_group  # Drift correction
   else:
       finalize_current_group()
       start_new_group(token)
   ```
   The running-mean anchor update is crucial: it handles the slight vertical drift that occurs in long rows where characters at the right edge sit a few pixels higher or lower than characters at the left edge.

5. **Left-to-right sorting:** Within each cluster, tokens are sorted by `x_min` (leftmost coordinate).

6. **Text concatenation:** Token texts are joined with single-space separators to produce `row.text`.

#### Why This Matters for Image Processing:
This stage converts the **2D spatial layout** of an image into **1D sequential text**, which is the fundamental transformation that makes regex-based parsing possible. Without accurate spatial reconstruction, tokens from different columns or adjacent rows would be interleaved, making downstream parsing impossible.

### 5.5 OCR Error Correction and Normalization

OCR engines — even state-of-the-art deep learning systems like PaddleOCR — produce systematic character-level errors. The normalization subsystem applies targeted corrections based on the visual similarity of characters.

#### 5.5.1 Date Normalization (3-Pass Pipeline)

**Pass 1 — OCR Typo Correction:**

Common OCR character confusions in date contexts:

| OCR Output | Correct Value | Visual Similarity Reason |
|:---|:---|:---|
| `2O2I` | `2021` | `O` resembles `0`; `I` resembles `1` |
| `l9` | `19` | Lowercase `l` resembles `1` |
| `I9` | `19` | Capital `I` resembles `1` |
| `0J` | `01` | `J` resembles `1` in some fonts |
| `?017` | `2017` | `?` resembles `2` in degraded scans |
| `1st`, `2nd`, `3rd` | `1`, `2`, `3` | Ordinal suffix removal |
| `13.10-2021` | `13-10-2021` | Mixed separator normalization |

**Pass 2 — Sequential Date Interpolation:**

When a row between two dated rows lacks a date, and the surrounding dates are 0–2 days apart, the missing date is interpolated:
```
Row i-1: "15-Jan-2021 ... 500.00"    → date = Jan 15
Row i:   "ATM WITHDRAWAL ... 200.00"  → date = MISSING → interpolated to Jan 15
Row i+1: "16-Jan-2021 ... 300.00"    → date = Jan 16
```

**Pass 3 — Date Canonicalization:**

All recognized dates are converted to ISO `YYYY-MM-DD` format using `dateutil.parser`. This prevents downstream date parsing failures caused by inconsistent formats:
- `01/04/2021` → `2021-04-01`
- `1 Apr 2021` → `2021-04-01`
- `01Aug` → `2025-08-01` (current year inferred)

#### 5.5.2 Amount Normalization

Monetary values undergo a similar correction pipeline:

| OCR Output | Correct Value | Correction Applied |
|:---|:---|:---|
| `530.C0` | `530.00` | `C` → `0` (letter-zero confusion) |
| `9200,C0` | `9200.00` | Comma decimal → dot; `C` → `0` |
| `-350 00` | `-350.00` | Whitespace decimal separator |
| `~8300.00` | `8300.00` | Leading noise stripping |
| `92.,50` | `92.50` | Duplicated separator cleanup |
| `925,90` | `925.90` | European comma-decimal normalization |

The amount normalizer includes critical safety guards:
- **Date overlap protection:** Amount candidates that overlap with date spans in the text are skipped to prevent corrupting dates (e.g., turning `13` in `13-10-2021` into `13.00`).
- **Merge guard:** If a candidate span already contains a valid `MONEY_RE` match, it is left untouched to prevent merging adjacent amounts.
- **Thousands guard:** A trailing 3-digit group after a comma is recognized as a thousands separator, not a decimal (e.g., `1,234` is not converted to `1.23`).

#### 5.5.3 Generic OCR Cleanup

The `ocr_cleanup.py` module provides domain-agnostic text cleaning:

- **Unicode normalization (NFC):** Canonicalizes composed/decomposed character forms.
- **Typographic replacement:** Curly quotes → straight quotes; em-dashes → hyphens; non-breaking spaces → regular spaces.
- **Whitespace collapse:** Multiple spaces/tabs → single space.
- **Noise character stripping:** Leading/trailing punctuation from table borders (`|`, `_`, etc.).

### 5.6 Layout Detection via Spatial Feature Analysis

The Layout Detector (`layout_detector.py`) identifies the structural organization of the transaction table using two complementary strategies:

#### Strategy 1: Header Row Detection

The detector scans reconstructed rows for column header keywords using a four-tier matching system:

1. **Exact alias match:** `"debit"` → `debit` (confidence 1.0)
2. **Word-level alias match:** Token containing `"debit"` as a word → `debit` (confidence × 0.85)
3. **Substring containment:** `"debitamount"` contains `"debit"` → `debit` (confidence × 0.80)
4. **Fuzzy matching (SequenceMatcher):** `"dehit"` ≈ `"debit"` (ratio ≥ 0.78) → `debit` (confidence × ratio × 0.75)

The fuzzy matching is essential for handling OCR-corrupted headers (e.g., `"Farticulars"` → `"Particulars"`, `"Balnnct"` → `"Balance"`).

**Header confidence scoring:**
```python
score = 0.25 + role_coverage × 0.45 + has_description × 0.10
        + has_money × 0.10 + avg_confidence × 0.25
```

A row must achieve confidence ≥ 0.40 and contain at least a `date` column plus one other target column to qualify as a header.

#### Strategy 2: Data-Row Geometric Inference (Fallback)

When no header row is detected, the system infers column positions from the data itself:

1. For each row, identify the **first date-valued token** (leftmost 3 tokens checked against `DATE_RE`).
2. Identify the **rightmost amount-valued token** (matched against `MONEY_RE`).
3. Identify the **first description token** (between date and amount, excluding monetary values).
4. Average the x-center positions across multiple qualifying rows to establish column centers.

Confidence scales with observation count: `confidence = min(0.70, 0.30 + observations × 0.06)`.

#### Column Role Assignment

Each detected column receives a normalized role (`date`, `description`, `debit`, `credit`, `balance`, `amount`) and a spatial position (x_min, x_max, x_center). This spatial information is critical for the semantic assignment stage.

### 5.7 Semantic Assignment via Geometric Distance Matching

The Semantic Assigner (`semantic_assigner.py`) maps extracted monetary values to their correct semantic fields using the detected table layout. This is fundamentally a **geometric image processing problem** — the system must determine which column each amount token belongs to based on its horizontal position in the original image.

#### Primary Method: Column-Order Matching
When the number of parsed amounts exactly matches the number of detected money columns, amounts are assigned left-to-right by x-position:

```
Layout: [Debit @ x=350] [Credit @ x=520] [Balance @ x=700]
Amounts: [500.00 @ x=345] [0.00 @ x=0] [12,500.00 @ x=695]
Result: debit=500.00, credit=null, balance=12,500.00
```

#### Fallback: Geometric Nearest-Column Assignment
When amounts don't match column count (common with OCR errors or optional columns), the system uses **minimum Euclidean distance matching** between each amount's x-center and each column's x-center:

```python
for amt_val, center in zip(amount_values, amount_centers):
    for role, col in money_columns:
        distances.append((abs(center - col.x_center), i, role, amt_val))
distances.sort(key=lambda x: x[0])  # Greedy closest match
```

This approach directly leverages the **spatial information preserved from the original image** through the bounding boxes.

#### Amount Token Localization

Finding the geometric position of an amount string within a row requires matching the amount text back to its source token. The system uses a three-pass token-matching strategy:

1. **Literal substring match:** Direct string containment (`"500.00"` in token text).
2. **Digit-sequence containment:** Compare only digit characters (handles separator corruption).
3. **Digit suffix match:** For split tokens (e.g., `"S1,"` + `"432.05"` → match on trailing digits).

---

## 6. Image Processing Techniques — Technical Deep Dive

### 6.1 Grayscale Conversion

```python
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
```

**Technique:** Single-channel luminance extraction using the standard weighted formula:
$$Y = 0.299R + 0.587G + 0.114B$$

**Rationale:** Bank statements are inherently monochromatic documents. Reducing to grayscale:
- Eliminates 67% of pixel data, reducing memory and computation.
- Removes color noise from mobile photographs (e.g., yellowish lighting cast).
- Provides the single-channel input required by subsequent thresholding operations.

### 6.2 Deskewing via Minimum Area Rectangle

```python
def _deskew(img):
    coords = np.column_stack(np.where(img < 128))  # Dark pixel coordinates
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img  # Skip negligible rotation
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

**Technique:** The algorithm:
1. **Thresholds** the image at intensity 128 to extract dark (text) pixel coordinates.
2. Fits a **minimum area bounding rectangle** to all dark pixels using `cv2.minAreaRect()`.
3. Extracts the rotation angle of this rectangle relative to the horizontal axis.
4. Applies an **affine rotation transformation** to correct the detected skew.

**Mathematical basis:** `cv2.minAreaRect()` computes the smallest enclosing rotated rectangle using the rotating calipers algorithm on the convex hull of the point set. The rotation angle of this rectangle reflects the dominant text orientation.

**Parameters:**
- Correction range: ±15° (angles beyond this suggest portrait/landscape confusion, not skew).
- Minimum correction: 0.5° (below this threshold, rotation artifacts exceed the skew itself).
- Interpolation: `INTER_CUBIC` for high-quality resampling.
- Border handling: `BORDER_REPLICATE` to avoid introducing black artifacts at rotated edges.

### 6.3 Non-Local Means Denoising

```python
def _denoise(img):
    return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
```

**Technique:** Non-Local Means (NLM) denoising, proposed by Buades, Coll, and Morel (2005). Unlike local filters (Gaussian blur, median filter) that average nearby pixels, NLM compares **patches** of pixels across a wider search area, averaging only pixels with similar local neighborhoods.

**Algorithm:**
For each pixel $p$, the denoised value is computed as a weighted average of all pixels $q$ within the search window:
$$\hat{I}(p) = \frac{1}{C(p)} \sum_{q \in S(p)} w(p,q) \cdot I(q)$$

where $w(p,q) = \exp\left(-\frac{\|P(p) - P(q)\|^2}{h^2}\right)$ and $P(p)$ is the patch centered at $p$.

**Parameters:**
- `h=10`: Filter strength. Controls the decay of the weight function. Higher values remove more noise but blur more detail.
- `templateWindowSize=7`: Size of the comparison patch (7×7 pixels). Captures enough local structure to distinguish text edges from noise.
- `searchWindowSize=21`: Size of the search area (21×21 pixels). Limits computational cost while still finding enough similar patches.

**Why NLM over simpler filters:**
- **Gaussian blur** destroys text edges, reducing OCR accuracy.
- **Median filter** is effective against salt-and-pepper noise but fails on Gaussian noise common in mobile photos.
- **NLM** preserves sharp text edges while removing both Gaussian and impulse noise, making it ideal for pre-OCR denoising.

### 6.4 Adaptive Gaussian Thresholding (Binarization)

```python
def _binarize(img):
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
```

**Technique:** Adaptive thresholding computes a separate threshold for each pixel based on the weighted mean of pixel values in a local neighborhood, minus a constant $C$:

$$T(x,y) = \text{GaussianWeightedMean}(I, \text{blockSize}) - C$$

A pixel is set to white (255) if its intensity exceeds $T(x,y)$, otherwise black (0).

**Why adaptive over global (Otsu's):**
Bank statement photographs frequently exhibit **uneven illumination** — one corner may be brighter due to the camera flash or ambient light. A global threshold (like Otsu's method) would under-binarize dark regions and over-binarize bright regions. Adaptive thresholding handles this by computing a spatially-varying threshold.

**Parameters:**
- `ADAPTIVE_THRESH_GAUSSIAN_C`: Uses a Gaussian-weighted sum of the neighborhood, giving more weight to nearby pixels. This produces smoother boundaries than the mean-based alternative.
- `blockSize=31`: The neighborhood size (31×31 pixels). Chosen to be larger than typical character stroke width (~5-10 px at 300 DPI) but smaller than the illumination gradient wavelength.
- `C=10`: The constant subtracted from the weighted mean. A positive value ensures that pixels near the mean (gray areas) are classified as background (white), which is correct for text documents where the background is lighter than the text.

### 6.5 Morphological Opening (Noise Cleanup)

```python
def _morphological_cleanup(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```

**Technique:** Morphological opening = erosion followed by dilation with the same structuring element.

1. **Erosion:** Each pixel is set to the minimum value in its neighborhood (defined by the kernel). This removes small bright noise specks that are smaller than the kernel.
2. **Dilation:** Each pixel is set to the maximum value in its neighborhood, restoring the size of text strokes that were shrunk by erosion.

**Structuring element:** A 2×2 rectangular kernel. This is intentionally small to:
- Remove 1-pixel noise specks (scanner dust, sensor noise).
- Preserve thin text strokes (critical for distinguishing characters like `i`, `l`, `1`).

**Mathematical formulation:**
$$\text{Opening}(I, K) = \text{Dilate}(\text{Erode}(I, K), K)$$

### 6.6 Canny Edge Detection and Hough Line Transform

The `detect_table_region()` function in the preprocessing module uses classical edge detection to locate the transaction table's physical boundaries:

```python
def detect_table_region(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=img.shape[1]//3, maxLineGap=20)
```

**Canny Edge Detection:**
1. Applies a Gaussian filter to smooth the image (aperture 3×3).
2. Computes image gradients using the Sobel operator in x and y directions.
3. Applies non-maximum suppression to thin edges to single-pixel width.
4. Applies hysteresis thresholding with low=50 and high=150 to connect strong edges and suppress weak isolated edges.

**Probabilistic Hough Line Transform (`HoughLinesP`):**
Detects line segments in the edge image. Parameters:
- `rho=1`: Distance resolution (1 pixel).
- `theta=π/180`: Angular resolution (1 degree).
- `threshold=100`: Minimum number of edge points to form a line.
- `minLineLength=width/3`: Lines must span at least 1/3 of the image width (filters short segments from text).
- `maxLineGap=20`: Gaps up to 20 pixels in a line are bridged (handles dashed table borders).

**Application:** The detected horizontal lines are clustered by y-coordinate. The densest band of horizontal lines typically corresponds to the ruled transaction table, providing a physical boundary for the transaction zone.

### 6.7 Affine Warp Transformation

Used in deskewing, the affine transformation applies a 2×3 transformation matrix to every pixel:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = M \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

where $M$ is the 2×3 rotation matrix computed by `cv2.getRotationMatrix2D()`:

$$M = \begin{bmatrix} \alpha & \beta & (1-\alpha) \cdot c_x - \beta \cdot c_y \\ -\beta & \alpha & \beta \cdot c_x + (1-\alpha) \cdot c_y \end{bmatrix}$$

with $\alpha = \text{scale} \cdot \cos(\theta)$ and $\beta = \text{scale} \cdot \sin(\theta)$.

`cv2.warpAffine()` applies this transformation using **bicubic interpolation** (`INTER_CUBIC`), which produces smooth results by fitting a cubic polynomial to the 4×4 neighborhood of each source pixel.

### 6.8 PDF Rasterization

```python
scale = dpi / 72  # PDF uses 72 points per inch
matrix = fitz.Matrix(scale, scale)
pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
```

**Technique:** Vector-to-raster conversion using the PyMuPDF rendering engine. The transformation matrix scales PDF coordinate space (72 PPI) to the target resolution (300 DPI), yielding a scale factor of 4.167×. The grayscale colorspace ensures single-channel output compatible with the OpenCV pipeline.

### 6.9 PaddleOCR's Internal Image Processing

PaddleOCR performs several internal image processing steps that are not directly controlled by the pipeline but significantly affect output quality:

1. **Image resizing:** Input images are resized to fit the detection network's expected dimensions while maintaining aspect ratio.
2. **Normalization:** Pixel values are normalized to [-1, 1] range for neural network input.
3. **Text region cropping:** Detected regions are perspective-corrected and cropped for recognition.
4. **Angle classification:** A lightweight CNN classifies each text region as 0° or 180° and applies correction.

---

## 7. Post-OCR Text Processing Pipeline

### 7.1 Row Reconstruction Algorithm

The row reconstruction algorithm converts the unordered set of OCR tokens (each with a bounding box) into ordered rows of text that mirror the visual layout of the document. This is the critical step that transforms 2D spatial information into 1D sequential text suitable for regex-based parsing.

**Complexity:** O(n log n) for sorting, O(n) for the greedy clustering pass, where n is the number of tokens.

**Edge case handling:**
- Empty token lists return empty row lists.
- Single-token documents produce a single row.
- The adaptive threshold (median height × 0.5) handles documents ranging from 72 DPI scans (small pixel coordinates) to 600 DPI scans (large pixel coordinates) without parameter tuning.

### 7.2 Date Normalization Pipeline

The date normalizer operates in three sequential passes:

1. **OCR Typo Correction:** Applies targeted character substitutions to date-like spans identified by `DATE_LIKE_RE`. Only characters within date-like contexts are modified, preventing corruption of non-date text.

2. **Sequential Date Interpolation:** For rows lacking dates that are sandwiched between rows with dates 0–2 days apart, a date is interpolated and prepended to the row text.

3. **Date Canonicalization:** All recognized dates are converted to `YYYY-MM-DD` format via `dateutil.parser`, using smart fallback for ambiguous bare `DD/MM` patterns.

### 7.3 Amount Normalization Pipeline

The amount normalizer applies corrections to monetary values using:

1. **Leading noise stripping:** Removes `~`, `$`, `₹`, `€`, `£` prefixes.
2. **OCR character substitution:** `O`→`0`, `C`→`0`, `I`→`1`, `l`→`1`, `+`→`4` — applied only when the character is adjacent to a digit.
3. **Separator normalization:** Fixes duplicated separators (`.,` → `.`), comma-as-decimal (`925,90` → `925.90`), and whitespace-as-decimal (`-350 00` → `-350.00`).
4. **Validation:** Every corrected amount is validated against `MONEY_RE`; invalid corrections are discarded.

### 7.4 Row Classification and Transaction Detection

Before attempting to parse transactions, rows pass through two pre-filters:

**RowClassifier:** A rule-based classifier that rejects non-transaction rows using regex patterns across 8 categories:
- Headers (`"Statement for Period"`, `"Account Number"`)
- Footers (`"Closing Balance"`, `"Thank You"`)
- Page markers (`"Page 1 of 5"`)
- Bank metadata (`"HDFC Bank"`, `"Savings Account"`)
- Summary sections (`"Reward Points"`, `"Account Summary"`)
- Disclaimers (`"Terms and Conditions"`)
- Column headers (3+ column name keywords in one row)
- Separators (rows of dashes, underscores, or equals signs)

**TransactionCandidateDetector:** Scores rows on positive signals:
- Date presence: +0.4
- ≥1 monetary value: +0.3
- ≥2 monetary values: +0.2
- Cr/Dr indicators: +0.2
- Transaction keywords (ATM, UPI, NEFT, etc.): +0.3
- Threshold: score ≥ 0.5 required.

### 7.5 Transaction Parsing with Stateful Date Inheritance

The Transaction Parser (`transaction_parser.py`) uses a **9-pattern date regex** (`DATE_RE`) that covers:

| Pattern | Format | Example |
|:---|:---|:---|
| 1 | DD/MM/YYYY, DD-MM-YYYY | `01/04/2021`, `15-03-2019` |
| 2 | DD-Mon-YYYY, DD Mon YYYY | `01-Apr-2021`, `15 Mar 2019` |
| 3 | DD Mon (no year) | `01 April`, `15 March` |
| 4 | DDMonYYYY (no spaces) | `01Apr2021`, `Jan2019` |
| 5 | DD.MM-YYYY, DD.MM.YYYY | `01.04-2021`, `15.03.2019` |
| 6 | YYYY-MM-DD (ISO) | `2021-04-01` |
| 7 | Mon DD YYYY, Mon DD, YYYY | `April 1, 2021`, `Mar 15 2019` |
| 8 | DDMon (no year) | `01Aug`, `15Mar` |
| 9 | MM/DD, DD/MM (bare) | `04/01`, `15/03` |

**Stateful Date Inheritance:** When a row contains amounts but no date, it inherits the `last_seen_date` from the most recently parsed transaction. This handles bank formats that group multiple transactions under a single date header.

**Row Merging:** When a row contains a date but no amount, and the next row contains amounts but no date, the two rows are merged — handling OCR-induced horizontal splits.

---

## 8. Quality Assurance and Image-Driven Validation

### 8.1 Low-Quality Image Detection Heuristic

The `BaseOCRBackend._is_low_quality()` method provides a fast image quality assessment:

```python
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
variance = float(np.var(img))
return variance < 1800
```

**Rationale:** The pixel intensity variance is a proxy for image contrast. Digital bank statements (sharp black text on white background) typically score 3,000–8,000. Low-contrast mobile photographs score below 1,800.

**Important finding:** While the heuristic correctly identifies low-quality images, the automatic preprocessing it triggered was counterproductive (see Section 5.2). The heuristic remains available but is no longer used to trigger preprocessing automatically.

### 8.2 Confidence Scoring Framework

Every stage of the pipeline produces confidence scores:

- **OCR confidence:** Per-token, from PaddleOCR's recognition network [0.0, 1.0].
- **Header match confidence:** Combined from alias match quality (0.55–1.0) and OCR token confidence, using a weighted formula: `confidence = match_conf × 0.65 + ocr_conf × 0.35`.
- **Layout detection confidence:** From header scoring or data-row observation count.
- **Semantic assignment confidence:** Average column confidence, scaled by 0.85 for geometric fallback.
- **Validation confidence:** Composite score with penalties for layout uncertainty, unassigned amounts, and validation issues.

### 8.3 Forensic Failure Analysis

The evaluation framework (`evaluation.py`) categorizes every failed row into one of six failure types:

| Failure Type | Description |
|:---|:---|
| `no_date_detected` | No date-like pattern found in the row |
| `date_parse_failed` | Date-like pattern found but rejected by validator |
| `insufficient_numeric_values` | Date found but no monetary values |
| `amount_extraction_failed` | Amounts found but could not be converted to float |
| `semantic_mismatch` | Amount count doesn't match layout column count |
| `unknown` | Unclassified failure |

Failure reports are anonymized (truncated to 50 characters) and written to `failure_reports.json` for post-hoc analysis.

---

## 9. Benchmark Results and Forensic Optimizations

### Dataset

- **91 bank statement documents** spanning diverse templates, layouts (multi-column tables, single-column), input qualities (digital scans, mobile photographs), and banks (Indian and international).

### Performance Summary

| Metric | Baseline | Final | Improvement |
|:---|:---|:---|:---|
| **Assigned Transactions** | 659 | **987** | **+328** (+49.8%) |
| **No Date Detected Failures** | 64 | **39** | **−25** (−39.1%) |
| **Date Parse Failures** | 32 | **19** | **−13** (−40.6%) |
| **Header Extraction Rate** | ~40% | **59.3%** | **+19.3%** |
| **Average Validation Confidence** | — | **77.03%** | — |
| **Average Layout Confidence** | — | **77.86%** | — |

### Major Optimizations and Their Impact

#### 1. Removing Automatic OpenCV Preprocessing (+256 Transactions)
The most impactful single change. The `_is_low_quality()` heuristic forced 36/91 statements through OpenCV binarization, which destroyed data in naturally legible photographs. Disabling this allowed PaddleOCR's superior internal preprocessing to handle these images natively.

#### 2. Eliminating Erroneous Classifier Rejections (+57 Transactions)
The RowClassifier incorrectly rejected rows containing `"Debit Card"` or `"Credit Card"` as metadata instead of transaction descriptions. Removing these overly aggressive rules recovered 57 valid transactions.

#### 3. Stateful Date Inheritance (+15 Transactions)
Implementing `last_seen_date` inheritance in the transaction parser recovered 15 transactions from bank formats where multiple transactions share a single date header.

#### 4. Unified DATE_RE Consolidation
Standardizing the date regex across all modules (previously each module had its own version) eliminated detection mismatches that caused layout boundaries to fail on valid edge cases.

---

## 10. Known Limitations

1. **Strict Date Regex Spacing:** The `DATE_RE` pattern requires a space between month abbreviations and single-digit days (e.g., `Jan 4` is valid, but `Jan4` may fail if not caught by Pattern 8). Relaxing this risks false positives.

2. **Extreme Document Skew:** While PaddleOCR's angle classifier handles 0°/180° rotation, extreme skew (>15°) causes bounding boxes to stair-step vertically, fracturing rows during reconstruction.

3. **Preprocessing Trade-off:** Disabling automatic preprocessing improved 22 statements but caused minor regressions (−1 to −3 transactions) on 3 statements. A future dual-pass architecture could address this.

4. **No Multi-Page Table Stitching:** Each page is processed independently. Transactions spanning page breaks are truncated.

5. **Micro-column Merging:** When amounts and dates lack sufficient whitespace, OCR may merge them into a single token (e.g., `12.2700`), causing parsing failures.

---

## 11. Technology Stack

| Component | Technology | Version | Role |
|:---|:---|:---|:---|
| OCR Engine | PaddleOCR | ≥ 2.7 | Text detection and recognition |
| Deep Learning Framework | PaddlePaddle | ≥ 2.6 | Neural network inference |
| Image Processing | OpenCV | ≥ 4.8, < 5.0 | Classical image enhancement |
| Array Computing | NumPy | ≥ 1.24, < 2.0 | Pixel data manipulation |
| Image I/O | Pillow | ≥ 10.0 | Image format handling |
| Date Parsing | python-dateutil | ≥ 2.8 | Robust date interpretation |
| PDF Rendering | PyMuPDF (fitz) | ≥ 1.23 | PDF → image conversion |
| Language | Python | 3.9+ | Core implementation |
| Testing | pytest | 9.0+ | 300+ unit and integration tests |

---

## 12. Testing Infrastructure

The pipeline is backed by a comprehensive test suite organized into four tiers:

- **Unit Tests** (`tests/unit/`): 12 test modules covering every pipeline component individually — amount normalizer, date normalizer, date regex, date validator, header extractor, layout detector, OCR cleanup, row classifier, semantic assigner, transaction detector, transaction parser, transaction reconstructor, and validator.

- **Integration Tests** (`tests/integration/`): Tests that verify cross-module data flow — normalize-then-parse, parse-then-assign.

- **Regression Tests** (`tests/regression/`): Tests specifically targeting previously observed failure modes — date parse failures, insufficient numeric values, no date detected, and unknown failures.

- **End-to-End Tests** (`tests/e2e/`): Full pipeline tests processing real statement images through every stage.

---

## 13. Conclusion

The BankStatementParser demonstrates that a carefully engineered, rules-based pipeline leveraging modern deep-learning OCR can achieve high transaction recovery rates on diverse, real-world bank statement documents **without relying on LLMs**. The key insight of this project is that the image processing pipeline must be designed with restraint — PaddleOCR's internal deep learning preprocessing outperforms aggressive classical preprocessing (OpenCV binarization) on the majority of inputs.

The system's image processing approach can be summarized across three layers:

1. **Deep Learning Layer (PaddleOCR):** Handles the hardest vision tasks — text detection in complex layouts, character recognition in degraded conditions, and orientation correction — using state-of-the-art neural networks (DB, CRNN, angle classifier).

2. **Classical Computer Vision Layer (OpenCV):** Provides optional enhancement for truly degraded inputs through a calibrated pipeline of deskewing, NLM denoising, adaptive thresholding, and morphological cleanup. Available on-demand but disabled by default after forensic analysis revealed it was counterproductive in most cases.

3. **Geometric Analysis Layer (Custom):** Converts 2D spatial information from bounding boxes into structured data through Y-axis row clustering, X-axis column detection, and geometric distance-based semantic assignment. This layer is entirely deterministic and auditable.

The +328 transaction improvement (+49.8%) achieved during the forensic optimization phase was accomplished entirely through **deterministic logic refinement** — no new ML models were trained. This underscores the importance of understanding the interaction between image processing stages and carefully measuring the impact of each intervention.

---

## 14. References

1. Buades, A., Coll, B., & Morel, J.-M. (2005). *A Non-Local Algorithm for Image Denoising.* IEEE CVPR.
2. Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020). *Real-time Scene Text Detection with Differentiable Binarization.* AAAI.
3. Shi, B., Bai, X., & Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition.* IEEE TPAMI.
4. PaddleOCR Documentation. https://github.com/PaddlePaddle/PaddleOCR
5. OpenCV Documentation. https://docs.opencv.org/
6. PyMuPDF Documentation. https://pymupdf.readthedocs.io/

---

*Report generated for BankStatementParser v1.0 | Dataset: 91 bank statement documents | November 2026*

