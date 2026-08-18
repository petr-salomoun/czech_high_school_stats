# Technical How-To Guide: Czech High School Stats Pipeline

This document explains the technical architecture, reproduction steps, CLI commands, and test workflows for the `czech_high_school_stats` (GymVA) analytical pipeline.

---

## 1. Prerequisites & Environment Setup

- **Python**: Version 3.11 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended on Windows)

### Installation

Clone the repository and set up a virtual environment:

```bash
git clone git@github.com:petr-salomoun/czech_high_school_stats.git
cd czech_high_school_stats

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .
```

Core dependencies:
- `pandas>=2.2`
- `numpy>=1.26`
- `openpyxl>=3.1`
- `scipy>=1.12`

---

## 2. Architecture: Two-Phase Workflow

The pipeline enforces a strict separation between external data fetching and deterministic local analysis:

```
[Phase 1: Archive Creation (Network)]
       CERMAT Open Data URLs
                │
                ▼
      Frozen Raw Archive (XLSX) + manifest.json (SHA256)
                │
                ▼
[Phase 2: Local Analysis (Strictly Offline)]
                ├── Normalized Tables (JPZ & MZ CSVs)
                ├── Cohort Panel Matching & Regressions
                └── Standalone HTML Report (archive_dashboard.html)
```

### Phase 1: Archive Creation (Network Allowed)

Downloads raw Excel datasets from official CERMAT endpoints into a timestamped directory and computes cryptographic checksums:

```bash
python -m gymnazium_value_added archive --archive-root data/archive
```

*(Alias: `python -m gymnazium_value_added archive-official --archive-root data/archive`)*

### Phase 2: Local Analysis (Strictly Offline)

Parses normalized tables, generates cross-year panels, fits school-level regression models, and produces the interactive dashboard without making any network requests:

```bash
python -m gymnazium_value_added analyze-local --archive data/archive/<freeze_id>
```

### Regenerating Dashboard Only

To re-render the interactive HTML dashboard from already-computed artifacts:

```bash
python -m gymnazium_value_added report-local --archive data/archive/<freeze_id>
```

### CLI Shortcut (`gymva` / `run`)

The CLI script `gymva` (or `python -m gymnazium_value_added run`) provides a single entry point:
- `gymva run --archive <freeze_dir>`: runs local analysis on an existing freeze.
- `gymva run --allow-network-fetch --archive-root data/archive`: explicitly performs archive creation and then local analysis.

---

## 3. Running the Test Suite

All unit and integration tests use standard Python `unittest`:

```bash
python -m unittest discover -s tests -p "test_*_unittest.py"
```

The test suite covers:
- URL discovery and validation (`test_discovery_unittest.py`)
- Ingestion and taxonomy normalization (`test_ingest_unittest.py`)
- Downloader and Excel validation (`test_io_unittest.py`)
- Archive freezing, parsing, and pipeline integration (`test_archive_unittest.py`)
- School-level regression and residual calculations (`test_analyze_unittest.py`)

---

## 4. Key Methodology & Parsing Semantics

1. **Programme Taxonomy**:
   Authoritative code-based classification (`GY4`, `GY6`, `GY8`, `LYC`, `SOS_*`, `SOU_*`, `NASTAVBA_*`).
2. **Gymnasium Cohort Matching**:
   Strict join on `school_key + programme_identity + component + graduation_year` with cohort lag `4/6/8` corresponding to programme duration.
   - Entrance cohorts 2017–2023 matched to graduation years 2021–2026.
3. **Model Specification for Expected vs. Observed MZ Scores**:
   - Regression: $\text{MZ score (\%)} \sim \text{JPZ published mean percentile} + \text{Cohort entry-year fixed effects}$
   - Weights: $\text{MZ candidates}$
   - Residuals: $\text{Observed MZ score} - \text{Expected MZ score}$ (in percentage points)
4. **Self-Contained Dashboard**:
   `report/archive_dashboard.html` bundles all required HTML, CSS, JavaScript, and embedded JSON data. It requires no web server and makes zero external CDN or API calls.

---

## 5. Directory Structure Overview

```
├── config/                          # Taxonomy rules, column mappings, sources config
├── data/
│   ├── manifest.json                # SHA256 checksums and provenance URLs
│   ├── source_override.json         # Relative source mapping
│   ├── raw/                         # 37 canonical CERMAT Excel workbooks (2016–2026)
│   └── processed/
│       ├── normalized/              # Cleaned component-level JPZ/MZ CSVs
│       └── reports/
│           ├── cohort_matched/      # Cohort panel, residuals, scenario intake CSVs
│           ├── cross_year_descriptive/ # Multi-year school trends CSVs
│           └── methodology.json     # Regression model coefficients and metadata
├── fixtures/                        # Mock data fixtures for test suite
├── gymnazium_value_added/           # Core package source code
│   ├── analyze.py                   # Regression fitting & value-added models
│   ├── archive_pipeline.py          # Two-phase archive freezing & parser
│   ├── archive_report.py            # Static standalone HTML dashboard generator
│   ├── cli.py                       # Command-line interface definitions
│   ├── config.py                    # Configuration loaders
│   ├── discovery.py                 # Source URL discovery heuristics
│   ├── ingest.py                    # Raw workbook parsers and schema normalizers
│   ├── io.py                        # Safe downloader with retry & validation
│   ├── model.py                     # Data dataclasses
│   └── report.py                    # Legacy tabular reporting
├── report/
│   └── archive_dashboard.html       # Standalone interactive HTML report
├── tests/                           # Unit test suite
├── pyproject.toml                   # Python packaging metadata
├── LICENSE                          # MIT License
└── README.md                        # Project introduction and overview
```
