from __future__ import annotations

import unittest
from pathlib import Path

from gymnazium_value_added.analyze import analyze_value_added
from gymnazium_value_added.ingest import normalize_admissions, normalize_maturita
from gymnazium_value_added.model import AnalysisConfig


class AnalyzeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_analysis_separates_residual_signal(self) -> None:
        admissions = normalize_admissions(
            source_path=self.root / "fixtures" / "admissions_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_adm",
        )
        maturita = normalize_maturita(
            source_path=self.root / "fixtures" / "maturita_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_mat",
        )

        result, meta = analyze_value_added(
            admissions,
            maturita,
            AnalysisConfig(cohort_lag_years=8, bootstrap_iterations=100, random_seed=7),
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]["school_name"], "Gymnázium Alfa")
        self.assertIn("repeated_cross_section_school_level", meta["analysis_type"])

    def test_analysis_metadata_mentions_school_level_metric_labels(self) -> None:
        admissions = normalize_admissions(
            source_path=self.root / "fixtures" / "admissions_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_adm",
        )
        maturita = normalize_maturita(
            source_path=self.root / "fixtures" / "maturita_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_mat",
        )

        _, meta = analyze_value_added(
            admissions,
            maturita,
            AnalysisConfig(cohort_lag_years=8, bootstrap_iterations=50, random_seed=7),
        )
        self.assertIn("school-level adjusted association model", meta["causal_warning"])


if __name__ == "__main__":
    unittest.main()
