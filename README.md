# Czech High School Admissions vs. Graduation Outcomes (Czech High School Stats)

Every spring in the Czech Republic, tens of thousands of eleven-, thirteen-, and fifteen-year-olds sit for the national standardized high school entrance exams (**Jednotná přijímací zkouška – JPZ**). Eight years later, students at academic grammar schools (*gymnázia*) sit for their standardized school-leaving examination (**Maturitní zkouška – MZ**).

Parents and students often wonder: *Which schools add the most value? Do top-scoring high schools merely admit the top 1% of applicants, or do they help their students progress faster? How selective are individual schools, and how do their graduating cohorts actually perform?*

This repository provides an open, auditable, and reproducible data pipeline analyzing official school-level aggregate data published by **CERMAT** (the Czech national assessment agency) and the **Ministry of Education, Youth and Sports (MŠMT)**. It tracks cohorts across time, models expected versus observed graduation outcomes, computes scenario intake selectivity proxies, and generates a self-contained interactive dashboard.

---

## The Czech High School System in Brief

For international readers and Czech families alike, here is how the academic high school tracks work:

- **8-year gymnázia (GY8)**: Students take the entrance exam at age ~11 after the 5th grade of primary school. They enter an intensive 8-year academic track and take their Maturita graduation exams at age ~19. This is one of the most selective educational paths in the Czech Republic.
- **6-year gymnázia (GY6)**: Students enter at age ~13 after the 7th grade for a 6-year academic track.
- **4-year gymnázia (GY4) & Vocational Schools (SOŠ / SOU)**: Students enter at age ~15 after completing 9th grade (basic school) for 4 years of secondary education.

Because 8-year gymnázia admit students at grade 5 and graduate them in year 8, tracking a cohort means pairing an entrance exam year $T$ with graduation year $T+8$ (for example, **entry cohort 2017 → graduation 2025**).

---

## The Data Sources

This project ingests and standardizes official public open data across multiple years:

1. **JPZ Entrance Exam Aggregates (2017–2026)**:
   - *Historic JPZ (2017–2023)*: School/programme level test results in Czech Language (**CJ**) and Mathematics (**M**), reporting candidate counts and published mean score percentiles across test-takers.
   - *Modern JPZ (2024–2026)*: Triplet format reporting applicant choices, capacity, and score distributions under the digitalized admissions system (Dipsy).
2. **MZ Maturita Graduation Exam Aggregates (2016–2026)**:
   - Programme-level spring and autumn graduation results for Czech Language (**CJ**), Mathematics (**M**), and English (**AJ**), including candidate headcounts, mean percentage scores, pass rates, and national school rankings.

---

## Example: Brno's 8-Year Gymnázia (Cohort 2017 → 2025)

To see how entrance selectivity and graduation outcomes connect, consider four premier 8-year gymnázia in Brno for the **2017 entrance cohort graduating in 2025**:

| School Name | REDIZO | JPZ Score Percentile | Synthetic Selectivity | MZ Mean Score (%) | MZ School Percentile | MZ Participation (CJ / M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Biskupské gymnázium a MŠ** | `600013405` | 60.7 | 87.94 | 88.56% | 98.62 | 100% / 42.6% |
| **Gymnázium Brno-Řečkovice** | `600013413` | 61.9 | 95.33 | 85.14% | 93.86 | 100% / 13.8% |
| **Gymnázium, tř. Kpt. Jaroše** | `600013481` | 61.6 | 98.05 | 89.07% | 99.19 | 100% / 70.0% |
| **Gymnázium Matyáše Lercha** | `600013626` | 61.7 | 99.61 | 91.54% | 99.87 | 100% / 53.8% |

### Understanding the Numbers & The Self-Selection Bias

1. **Entrance Test-Taker Pool vs. Admitted Pool**:
   In the raw CERMAT data, the published *JPZ Percentile* (~61–62 for all four schools above) represents the average score of all students who sat the exam at that school—not just those admitted. Because top gymnázia attract high-scoring applicants but admit only the top slice, we compute a **Synthetic Selectivity Proxy** (87.9% to 99.6%). This models what score threshold an admitted class represents given candidate volume and graduating class size.

2. **Graduation Performance (MZ)**:
   All four schools rank at the top of national graduation percentiles (93rd to 99th+ percentile nationally in spring Maturita).

3. **The Math Participation Trap (Self-Selection)**:
   In the Czech Maturita exam, Czech Language (**CJ**) is mandatory for 100% of students, but Mathematics (**M**) is optional (students choose between Math and a Foreign Language).
   - At **Gymnázium Brno-Řečkovice**, only **13.8%** of the graduating cohort elected to take the Math Maturita. This means only a small self-selected group of math enthusiasts took the test.
   - At **Gymnázium, tř. Kpt. Jaroše** (famed for its mathematical focus), **70.0%** of the entire cohort took the Math Maturita.
   - Comparing raw math averages directly without noting participation rates would distort reality: getting an 85% average when 14% of your top math students sit the exam is fundamentally different from achieving an 89% average when 70% of the entire student body takes it.
   - That is why our dashboard and reports explicitly include the **MZ participation rate (relative to CJ)** column.

---

## Important Methodological Caveats

When exploring the data and models in this project, keep the following guardrails in mind:

- **Descriptive Associations, Not Causal Claims**: School-level regressions reflect aggregate descriptive associations. They cannot prove causal "school value-added" or isolate teacher effectiveness from student socio-economic background, tutoring, peer effects, or unobserved student motivation.
- **Aggregate Public Data Only**: All calculations rely strictly on published school/programme-level tables. No individual student records exist in the public domain.
- **Test-Takers vs. Unique Applicants**: Historic JPZ candidate numbers represent exam seatings (which could be up to two per applicant) rather than unique persons. Historical accepted/rejected lists and student preference orderings are not published in CERMAT historical archives.
- **Scenario Selectivity is a Model Proxy**: The synthetic intake selectivity metric is an algorithmic estimate based on capacity throughput and score distributions under explicit scenario assumptions, not observed administrative cutoffs.

---

## What's in This Repository

```
czech_high_school_stats/
├── report/
│   └── archive_dashboard.html       # Main Deliverable: standalone interactive HTML dashboard
├── data/
│   ├── raw/                         # 37 original canonical CERMAT Excel workbooks (2016–2026)
│   ├── processed/                   # Standardized CSVs & JSON regression metadata
│   │   ├── normalized/              # Cleaned component-level JPZ and MZ tables
│   │   └── reports/                 # Cohort panels, expected-vs-observed residuals, trends
│   ├── manifest.json                # SHA256 checksums and provenance URLs for all raw sources
│   └── source_override.json         # Relative source mapping
├── gymnazium_value_added/           # Core Python analytical engine & CLI tool
├── tests/                           # Full unit and integration test suite
├── config/                          # Taxonomy and column normalization rules
├── fixtures/                        # Test fixtures
├── pyproject.toml                   # Packaging and dependency declarations
├── TECHNICAL_HOWTO.md               # Step-by-step reproduction and CLI guide
└── LICENSE                          # MIT License
```

### Main Deliverable: Interactive HTML Dashboard

The primary visual deliverable is located at [`report/archive_dashboard.html`](report/archive_dashboard.html).

It is a zero-dependency, self-contained interactive dashboard containing:
- **Global School Search & Filter**: Filter by school name, city, address, REDIZO, or school type.
- **Cohort Scatter Matrix**: Interactive plots comparing entrance percentiles to graduation school-mean percentile ranks (0–100 scale).
- **Observed vs. Model-Expected Outcomes**: School-level residual regressions (`MZ score ~ JPZ published mean percentile + cohort fixed effect`) weighted by candidate volume.
- **Scenario Intake Selectivity**: Interactive exploration of selectivity proxies across CJ, M, and AJ outcomes.
- **Cross-Year Historic Trends**: Time series tracking school performance across all available examination years.

---

## How to Explore & Reproduce

To run the data pipeline locally, regenerate the dashboard, or run the test suite, please see the step-by-step instructions in **[TECHNICAL_HOWTO.md](TECHNICAL_HOWTO.md)**.

Quick start:
```bash
# Clone the repository
git clone git@github.com:petr-salomoun/czech_high_school_stats.git
cd czech_high_school_stats

# Install in a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the test suite
python -m unittest discover -s tests -p "test_*_unittest.py"
```

---

## Data Attribution & License

- **Data Sources**: Official open datasets published by **CERMAT** (Centrum pro zjišťování výsledků vzdělávání) and **MŠMT** (Ministerstvo školství, mládeže a tělovýchovy ČR).
- **Code & Pipeline License**: This project is licensed under the [MIT License](LICENSE) © 2026 Petr Salomoun.
