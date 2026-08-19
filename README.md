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

![Brno GY8 Entrance vs Graduation Outcomes](report/images/brno_gy8_jpz_vs_mz.png)

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

![Brno GY8 Math Maturita Participation Rates](report/images/brno_gy8_math_participation.png)

*(Note: For full statistical methodology, modeling details, and limitations, please see the caveats section in [TECHNICAL_HOWTO.md](TECHNICAL_HOWTO.md).)*

---

## Interactive HTML Dashboard

Explore the full dataset interactively:

👉 **[Open the live interactive dashboard](https://petr-salomoun.github.io/czech_high_school_stats/)**

*Tip for offline or local use*: The dashboard is also available as a self-contained static file in the repository at [`report/archive_dashboard.html`](report/archive_dashboard.html) (and [`docs/index.html`](docs/index.html)). Because GitHub's web UI blob viewer does not render large HTML files directly, we recommend viewing via the live GitHub Pages link above, or cloning/downloading `report/archive_dashboard.html` to open directly in any web browser (zero external CDN or server dependencies required).

---

## Technical Documentation & Reproduction

For step-by-step instructions on running the pipeline locally, CLI usage, full methodological caveats, regression specifications, and repository architecture, see **[TECHNICAL_HOWTO.md](TECHNICAL_HOWTO.md)**.

---

## License & Attribution

- **Code & Original Analysis**: Licensed under the [MIT License with Attribution Requirement](LICENSE) © 2026 Petr Salomoun. Any public use, publication, redistribution, or derivative work based on this software or its outputs must include clear attribution to the original author, Petr Salomoun, and a link to this repository.
- **Underlying Exam Data**: Official open datasets published by **CERMAT** and **MŠMT**, subject to Czech public sector open data terms. See [DATA_LICENSE.md](DATA_LICENSE.md) for data provenance and reuse terms.
