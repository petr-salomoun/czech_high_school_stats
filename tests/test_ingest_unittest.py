from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from gymnazium_value_added.ingest import (
    filter_eight_year_programs,
    normalize_admissions,
    normalize_jpz_historic_results,
    normalize_maturita,
)
from gymnazium_value_added.archive_pipeline import parse_jpz_historic_components


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_normalize_and_filter_admissions(self) -> None:
        adm = normalize_admissions(
            source_path=self.root / "fixtures" / "admissions_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_adm",
        )
        self.assertTrue({"school_id", "applications", "capacity", "source_row_number"}.issubset(set(adm.columns)))
        filtered = filter_eight_year_programs(adm)
        self.assertEqual(len(filtered), len(adm))

    def test_filter_eight_year_programs_handles_missing_kkov(self) -> None:
        df = pd.DataFrame(
            [
                {"school_id": "1", "school_name": "A", "year": 2018, "program_name": "Gymnázium osmileté"},
                {"school_id": "2", "school_name": "B", "year": 2018, "program_name": "Jiné"},
            ]
        )
        filtered = filter_eight_year_programs(df)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["school_id"], "1")

    def test_normalize_maturita(self) -> None:
        mat = normalize_maturita(
            source_path=self.root / "fixtures" / "maturita_fixture.csv",
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="fixture_mat",
        )
        self.assertTrue({"school_id", "subject", "candidates", "mean_score", "source_row_number"}.issubset(set(mat.columns)))
        self.assertTrue(bool(mat["mean_score"].notna().all()))

    def test_normalize_wide_maturita_aggregate_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MZ2026j_SC_skolobory.xlsx"
            pd.DataFrame(
                {
                    "TŘÍDĚNÍ": ["redizo", "redizo_smo16"],
                    "REDIZO": ["600000001", "600000001"],
                    "NÁZEV ŠKOLY": ["S1", "S1"],
                    "ADRESA ŠKOLY": ["Ulice 1, Praha 11000", "Ulice 1, Praha 11000"],
                    "ROK": [2026, 2026],
                    "SMO16": ["CELKEM", "GY8"],
                    "CELKEM KONALI": [120, 10],
                    "CELKEM PRŮMĚRNÝ % SKÓR": [88.0, 74.0],
                    "CELKEM PODÍL ÚSPĚŠNÝCH": [0.99, 0.90],
                    "ČJ KONALI": [110, 10],
                    "ČJ PRŮMĚRNÝ % SKÓR": [89.0, 75.0],
                    "ČJ PODÍL ÚSPĚŠNÝCH": [0.99, 0.91],
                    "M KONALI": [108, 10],
                    "M PRŮMĚRNÝ % SKÓR": [87.0, 73.0],
                    "M PODÍL ÚSPĚŠNÝCH": [0.98, 0.89],
                    "AJ KONALI": [100, 10],
                    "AJ PRŮMĚRNÝ % SKÓR": [90.0, 76.0],
                    "AJ PODÍL ÚSPĚŠNÝCH": [0.99, 0.92],
                }
            ).to_excel(path, sheet_name="2026", index=False)
            mat = normalize_maturita(
                source_path=path,
                mapping_path=self.root / "config" / "column_mappings.json",
                source_id="maturita_2026",
            )
            self.assertEqual(set(mat["subject"].tolist()), {"TOTAL", "CJ", "M", "AJ"})
            one = mat[mat["school_id"].astype(str).eq("600000001")]
            cj_vals = sorted(one[one["subject"] == "CJ"]["candidates"].astype(int).tolist())
            total_vals = sorted(one[one["subject"] == "TOTAL"]["candidates"].astype(int).tolist())
            self.assertIn(10, cj_vals)
            self.assertIn(10, total_vals)

    def test_normalize_historic_jpz_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "JPZ2018_red.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame(
                    {
                        "REDIZO / KRAJ / OBOROVÁ SKUPINA": ["600000010"],
                        "OBOROVÁ SKUPINA": ["Gymnázium osmileté"],
                        "ROČNÍK": [5],
                        "NÁZEV ŠKOLY": ["Gymnazium H"],
                        "ČESKÝ JAZYK": [72],
                        "MATEMATIKA": [70],
                    }
                ).to_excel(writer, sheet_name="JPZ2018_red", index=False)
            hist = normalize_jpz_historic_results(
                source_path=path,
                mapping_path=self.root / "config" / "column_mappings.json",
                source_id="hist_jpz_2018",
            )
            self.assertTrue({"selection_metric", "selection_metric_observed", "selection_metric_method"}.issubset(set(hist.columns)))
            self.assertTrue(hist["selection_metric"].isna().all())
            self.assertEqual(hist.iloc[0]["selection_metric_method"], "historic_jpz_results_only")
            self.assertEqual(float(hist.iloc[0]["avg_admission_score"]), 71.0)
            self.assertEqual(int(hist.iloc[0]["year"]), 2018)

    def test_parse_jpz_historic_classifies_all_types_from_official_code_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "JPZ2019_types.xlsx"
            pd.DataFrame(
                {
                    "REDIZO / KRAJ / OBOROVÁ SKUPINA": ["600100001", "600100002", "600100003", "600100004"],
                    "OBOROVÁ SKUPINA": ["79-41-K/81", "79-41-K/61", "79-41-K/41", "NEZNAMY"],
                    "ROČNÍK": [5, 7, 9, 6],
                    "NÁZEV ŠKOLY": ["A", "B", "C", "D"],
                    "ADRESA ŠKOLY": ["U 1, Praha 11000", "U 2, Brno 60200", "U 3, Olomouc 77900", "U 4, Ostrava 70200"],
                    "ČJL průměrné percentilové umístění": [60.0, 58.0, 56.0, 54.0],
                    "MAT průměrné percentilové umístění": [59.0, 57.0, 55.0, 53.0],
                }
            ).to_excel(path, index=False)
            out = parse_jpz_historic_components(path, 2019, "jpz_2019_historic_vysledky", _force_single_header=True)
            self.assertFalse(out.empty)
            self.assertTrue({"school_type", "programme_taxonomy", "programme_identity", "classification_quality", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "entrant_grade", "grade"}.issubset(set(out.columns)))
            self.assertIn("GY8", set(out["school_type"].astype(str).unique().tolist()))
            self.assertIn("GY6", set(out["school_type"].astype(str).unique().tolist()))
            self.assertIn("GY4", set(out["school_type"].astype(str).unique().tolist()))
            self.assertIn("UNKNOWN", set(out["school_type"].astype(str).unique().tolist()))

    def test_normalize_real_maturita_2026_if_available(self) -> None:
        path = self.root / "data" / "raw" / "maturita_2026.xlsx"
        if not path.exists():
            self.skipTest("local real-file fixture unavailable")
        mat = normalize_maturita(
            source_path=path,
            mapping_path=self.root / "config" / "column_mappings.json",
            source_id="maturita_2026_j",
        )
        self.assertFalse(mat.empty)
        self.assertTrue({"CJ", "M"}.issubset(set(mat["subject"].astype(str).unique().tolist())))
        self.assertTrue(mat["school_id"].astype(str).str.fullmatch(r"\d+").any())


if __name__ == "__main__":
    unittest.main()
