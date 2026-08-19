# Technical How-To Guide: Czech High School Stats Pipeline

This document explains the technical architecture, reproduction steps, CLI commands, methodological caveats, directory layout, and test workflows for the `czech_high_school_stats` (GymVA) analytical pipeline.

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
- `matplotlib>=3.8`

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

## 3. How to Explore & Reproduce

To reproduce the analysis, regenerate data tables, or execute the end-to-end pipeline locally:

```bash
# Clone the repository
git clone git@github.com:petr-salomoun/czech_high_school_stats.git
cd czech_high_school_stats

# Set up virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run local analysis and regenerate reports
python -m gymnazium_value_added analyze-local
```

All output CSV panels and JSON summaries will be written to `data/processed/reports/`, and the interactive HTML report will be updated in `report/archive_dashboard.html`.

---

## 4. Running the Test Suite

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

## 5. Key Methodology & Parsing Semantics

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

## 6. Important Methodological Caveats

When exploring the data and models in this project, keep the following guardrails in mind:

- **Descriptive Associations, Not Causal Claims**: School-level regressions reflect aggregate descriptive associations. They cannot prove causal "school value-added" or isolate teacher effectiveness from student socio-economic background, tutoring, peer effects, or unobserved student motivation.
- **Aggregate Public Data Only**: All calculations rely strictly on published school/programme-level tables. No individual student records exist in the public domain.
- **Test-Takers vs. Unique Applicants**: Historic JPZ candidate numbers represent exam seatings (which could be up to two per applicant) rather than unique persons. Historical accepted/rejected lists and student preference orderings are not published in CERMAT historical archives.
- **Scenario Selectivity is a Model Proxy**: The synthetic intake selectivity metric is an algorithmic estimate based on capacity throughput and score distributions under explicit scenario assumptions, not observed administrative cutoffs.

---

## 7. What's in This Repository (Directory Structure)

```
czech_high_school_stats/
├── report/
│   ├── archive_dashboard.html       # Standalone interactive HTML dashboard (zero-dependency)
│   └── images/                      # Generated cohort visualization charts (PNG)
│       ├── brno_gy8_jpz_vs_mz.png
│       └── brno_gy8_math_participation.png
├── data/
│   ├── raw/                         # 37 canonical CERMAT Excel workbooks (2016–2026)
│   ├── processed/                   # Standardized CSVs & JSON regression metadata
│   │   ├── normalized/              # Cleaned component-level JPZ and MZ tables
│   │   └── reports/                 # Cohort panels, expected-vs-observed residuals, trends
│   ├── manifest.json                # SHA256 checksums and provenance URLs for all raw sources
│   └── source_override.json         # Relative source mapping
├── gymnazium_value_added/           # Core Python analytical engine & CLI tool
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
├── tests/                           # Full unit and integration test suite
├── config/                          # Taxonomy and column normalization rules
├── fixtures/                        # Test fixtures
├── pyproject.toml                   # Packaging and dependency declarations
├── TECHNICAL_HOWTO.md               # Step-by-step reproduction and CLI guide
├── DATA_LICENSE.md                  # Data provenance & public sector licensing terms
└── LICENSE                          # MIT License with Attribution Requirement
```

---

## 8. Data Attribution & Licensing

- **Code & Analysis**: Licensed under the [MIT License with Attribution Requirement](LICENSE) © 2026 Petr Salomoun.
- **Source Data**: Official public open data published by CERMAT and MŠMT. See [DATA_LICENSE.md](DATA_LICENSE.md).
