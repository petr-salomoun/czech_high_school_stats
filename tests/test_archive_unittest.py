from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import pandas as pd

from gymnazium_value_added.archive_pipeline import analyze_archive_local, create_archive
from gymnazium_value_added.archive_report import write_archive_dashboard
from gymnazium_value_added.cli import main
from gymnazium_value_added.report import write_report


class _DashboardScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.executable_dashboard_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attributes = dict(attrs)
        if attributes.get("id") != "dashboard-js":
            return
        script_type = (attributes.get("type") or "text/javascript").lower()
        self.executable_dashboard_script = script_type in {
            "text/javascript",
            "application/javascript",
            "module",
        }


class ArchiveLocalTests(unittest.TestCase):
    def _scenario_uplift_expected(self, p: float, dispersion: float = 0.75) -> float:
        if p >= 1.0:
            return 0.0
        z = NormalDist().inv_cdf(1.0 - p)
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return dispersion * phi / p

    def test_report_table_shows_address_and_omits_internal_key_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = write_report(
                pd.DataFrame(
                    [
                        {
                            "school_name": "Gymnázium A",
                            "address": "Ulice 1, Praha 11000",
                            "school_id": "600100001",
                            "mean_selection_metric": 12.3,
                            "mean_admission_score": 45.6,
                            "observed_outcome": 78.9,
                            "expected_outcome": 70.0,
                            "value_added": 8.9,
                            "value_added_ci_low": 4.2,
                            "value_added_ci_high": 13.6,
                            "quality_flag": "ok",
                        }
                    ]
                ),
                {"methodology": "x", "outcome_definition": "mean_score"},
                tmp,
            )
            html_text = Path(out["html"]).read_text(encoding="utf-8")
            self.assertIn("Address", html_text)
            self.assertNotIn("school_key", html_text)

    def test_dashboard_definitions_use_safe_json_string_quoting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            with patch("gymnazium_value_added.archive_report._build_model", return_value={}):
                dashboard = write_archive_dashboard(archive_dir)

            script = dashboard.read_text(encoding="utf-8")
            self.assertIn('const defs=["Selectivity percentile (synthetic)', script)
            self.assertIn("MZ throughput proxy (max candidates)", script)
            self.assertNotIn("school\\'s", script)

    def test_dashboard_hides_raw_category_and_quality_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = Path(tmp)
            with patch("gymnazium_value_added.archive_report._build_model", return_value={}):
                dashboard = write_archive_dashboard(archive_dir)

            html_text = dashboard.read_text(encoding="utf-8")
            self.assertNotIn("KKOV", html_text)
            self.assertNotIn("programme", html_text)
            self.assertNotIn("quality_flag", html_text)

    def _mk_historic_multiheader_fixture(self, path: Path) -> None:
        rows = [
            ["PŘEHLED", "", "", "", "", "", "", "", "", "", "", ""],
            [
                "",
                "REDIZO / KRAJ / OBOROVÁ SKUPINA",
                "OBOROVÁ SKUPINA",
                "ROČNÍK",
                "NÁZEV ŠKOLY",
                "ADRESA ŠKOLY",
                "ČJL",
                "",
                "",
                "MAT",
                "",
                "",
            ],
            [
                "",
                "REDIZO",
                "",
                "",
                "",
                "",
                "PŘIHLÁŠENI",
                "KONALI",
                "PRŮMĚRNÉ PERCENTILOVÉ UMÍSTĚNÍ",
                "PŘIHLÁŠENI",
                "KONALI",
                "PRŮMĚRNÉ PERCENTILOVÉ UMÍSTĚNÍ",
            ],
            [
                "",
                "600199999",
                "GY8",
                5,
                "Gymnázium Multi",
                "Ulice 9, Praha 11000",
                120,
                110,
                62.5,
                118,
                108,
                59.5,
            ],
            [
                "",
                "600188888",
                "GY8",
                5,
                "Gymnázium Multi 2",
                "Ulice 8, Brno 60200",
                90,
                80,
                52.5,
                95,
                85,
                49.5,
            ],
        ]
        pd.DataFrame(rows).to_excel(path, index=False, header=False)

    def _mk_sources(self, root: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}

        hist_cols = {
            "REDIZO / KRAJ / OBOROVÁ SKUPINA": ["600100001", "600100002", "600100003"],
            "OBOROVÁ SKUPINA": ["79-41-K/81", "79-41-K/61", "79-41-K/41"],
            "ROČNÍK": [5, 7, 9],
            "NÁZEV ŠKOLY": ["Gymnázium A", "Gymnázium B", "Gymnázium C"],
            "ADRESA ŠKOLY": ["Ulice 1, Praha 11000", "Ulice 2, Brno 60200", "Ulice 3, Olomouc 77900"],
            "ČJL průměrné percentilové umístění": [60.0, 50.0, 47.0],
            "MAT průměrné percentilové umístění": [55.0, 45.0, 44.0],
        }
        for y in range(2017, 2024):
            p = root / f"JPZ{y}_skoly-skolobory_vysledky.xlsx"
            pd.DataFrame(hist_cols).to_excel(p, index=False)
            files[f"jpz_{y}_historic_vysledky"] = p

        for y in range(2024, 2027):
            app = root / f"PZ{y}_kolo1_skolobory_prihlasky.xlsx"
            cap = root / f"PZ{y}_kolo1_skolobory_kapacity.xlsx"
            res = root / f"PZ{y}_kolo1_skolobory_vysledky.xlsx"
            pd.DataFrame(
                {
                    "REDIZO": ["600100001", "600100002"],
                    "NÁZEV ŠKOLY": ["Gymnázium A", "Gymnázium B"],
                    "ADRESA ŠKOLY": ["Ulice 1, Praha 11000", "Ulice 2, Brno 60200"],
                    "SMO16": ["GY8", "GY8"],
                    "PŘIHLÁŠKY": [120, 100],
                }
            ).to_excel(app, index=False)
            pd.DataFrame(
                {
                    "REDIZO": ["600100001", "600100002"],
                    "NÁZEV ŠKOLY": ["Gymnázium A", "Gymnázium B"],
                    "ADRESA ŠKOLY": ["Ulice 1, Praha 11000", "Ulice 2, Brno 60200"],
                    "SMO16": ["GY8", "GY8"],
                    "KAPACITA": [30, 25],
                }
            ).to_excel(cap, index=False)
            pd.DataFrame(
                {
                    "REDIZO": ["600100001", "600100002"],
                    "PRŮMĚRNÝ % SKÓR": [72.0, 67.0],
                }
            ).to_excel(res, index=False)
            files[f"jpz_{y}_kolo1_prihlasky"] = app
            files[f"jpz_{y}_kolo1_kapacity"] = cap
            files[f"jpz_{y}_kolo1_vysledky"] = res

        for y in range(2015, 2027):
            for v in ("j", "jap"):
                p = root / f"MZ{y}{v}_SC_skolobory.xlsx"
                df = pd.DataFrame(
                    {
                        "TŘÍDĚNÍ": [
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                            "redizo_smo16",
                        ],
                        "REDIZO": [
                            "600100001", "600100002", "600100003", "600100004", "600100005", "600100006", "600100007", "600100008", "600100009",
                            "600100010", "600100011", "600100012", "600100013", "600100014", "600100015", "600100016", "600100017", "600100018",
                        ],
                        "NÁZEV ŠKOLY": [
                            "Gymnázium A", "Gymnázium B", "Gymnázium C", "Lyceum D", "SOŠ tech E", "SOŠ ekonom F", "SOŠ hotel G", "SOŠ human H", "SOŠ agri I",
                            "SOŠ zdrav J", "SOŠ uměn K", "SOU tech L", "SOU M", "Nástavba tech N", "Nástavba O", "Souhrn P", "Neznámá Q", "SOŠ tech R",
                        ],
                        "ADRESA ŠKOLY": [
                            "Ulice 1, Praha 11000", "Ulice 2, Brno 60200", "Ulice 3, Olomouc 77900", "Ulice 4, Ostrava 70200", "Ulice 5, Plzeň 30100", "Ulice 6, Liberec 46001",
                            "Ulice 7, Zlín 76001", "Ulice 8, Pardubice 53002", "Ulice 9, Hradec Králové 50003", "Ulice 10, Jihlava 58601", "Ulice 11, Ústí nad Labem 40001",
                            "Ulice 12, Tábor 39001", "Ulice 13, Kladno 27201", "Ulice 14, Most 43401", "Ulice 15, Cheb 35002", "Ulice 16, Písek 39701", "Ulice 17, Třebíč 67401", "Ulice 18, Děčín 40502",
                        ],
                        "ROK": [y] * 18,
                        "SMO16": ["GY8", "GY6", "GY4", "LYC", "ST1", "SEK", "SHP", "SHU", "SZE", "SZD", "SUM", "UTE", "UOS", "NTE", "NOS", "CELKEM", "NEZNAMY", "ST2"],
                        "KKOV": ["79-41-K/81", "79-41-K/61", "79-41-K/41", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                        "NÁZEV OBORU": ["Gymnázium", "Gymnázium", "Gymnázium", "Lyceum", "Strojírenství", "Ekonomika", "Hotelnictví", "Humanitní", "Agro", "Zdravotnictví", "Umění", "Technické učiliště", "Učiliště", "Nástavba technická", "Nástavba", "Součet", "Nezařazeno", "Elektro"],
                        "ZAMĚŘENÍ": ["8leté", "6leté", "4leté", "všeobecné", "tech", "ekon", "hotel", "human", "agro", "zdrav", "arts", "ute", "uos", "nte", "nos", "all", "unknown", "tech2"],
                        "CELKEM KONALI": [30, 25, 20, 18, 22, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7],
                        "CELKEM PRŮMĚRNÝ % SKÓR": [72.0 + (y - 2025), 68.0 + (y - 2025), 65.0 + (y - 2025), 63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0, 49.0],
                        "CELKEM PODÍL ÚSPĚŠNÝCH": [0.92, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74],
                        "ČJ KONALI": [30, 25, 20, 18, 22, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7],
                        "ČJ PRŮMĚRNÝ % SKÓR": [73.0 + (y - 2025), 69.0 + (y - 2025), 66.0 + (y - 2025), 64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0],
                        "ČJ PODÍL ÚSPĚŠNÝCH": [0.93, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75],
                        "M KONALI": [30, 25, 20, 18, 22, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7],
                        "M PRŮMĚRNÝ % SKÓR": [71.0 + (y - 2025), 67.0 + (y - 2025), 64.0 + (y - 2025), 62.0, 61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0, 49.0, 48.0],
                        "M PODÍL ÚSPĚŠNÝCH": [0.90, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72],
                        "AJ KONALI": [30, 25, 20, 18, 22, 21, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7],
                        "AJ PRŮMĚRNÝ % SKÓR": [74.0 + (y - 2025), 70.0 + (y - 2025), 67.0 + (y - 2025), 65.0, 64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0],
                        "AJ PODÍL ÚSPĚŠNÝCH": [0.94, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76],
                    }
                )
                df.to_excel(p, index=False)
                files[f"mz_{y}_{v}"] = p

        return files

    def test_archive_manifest_and_local_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")

            archive_dir = create_archive(root / "archive", source_override=ov)
            manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
            src = manifest["sources"]
            self.assertTrue(any(s.get("source_id") == "jpz_2016_historic_vysledky" and s.get("status") == "unavailable" for s in src))
            self.assertTrue(any(s.get("source_id") == "jpz_2017_historic_vysledky" and s.get("status") == "downloaded" for s in src))
            self.assertTrue(any(s.get("source_id") == "mz_2026_jap" and s.get("status") == "downloaded" for s in src))

            with patch("gymnazium_value_added.archive_pipeline.download_url", side_effect=AssertionError("network forbidden")):
                result = analyze_archive_local(archive_dir)
            self.assertTrue(Path(result["normalized_dir"]).exists())
            self.assertTrue((Path(result["normalized_dir"]) / "jpz_components.csv").exists())
            self.assertTrue((Path(result["normalized_dir"]) / "mz_components.csv").exists())
            self.assertTrue((Path(result["normalized_dir"]) / "school_dimension.csv").exists())
            self.assertTrue((Path(result["reports_dir"]) / "cohort_matched" / "cohort_component_panel.csv").exists())
            self.assertTrue((Path(result["reports_dir"]) / "cross_year_descriptive" / "jpz_school_component_trends.csv").exists())
            self.assertTrue((Path(result["reports_dir"]) / "archive_dashboard.html").exists())

            rc = main(["report-local", "--archive", str(archive_dir)])
            self.assertEqual(rc, 0)

            expected_dash = archive_dir / "reports" / "archive_dashboard.html"
            self.assertTrue(expected_dash.exists())
            self.assertEqual(sorted(str(p) for p in archive_dir.rglob("archive_dashboard.html")), [str(expected_dash)])

            dash = expected_dash.read_text(encoding="utf-8")
            self.assertIn("Selectivity percentile (synthetic)", dash)
            self.assertIn("MZ throughput proxy (max candidates)", dash)
            self.assertIn('data-tab="summary"', dash)
            self.assertIn('data-tab="cohort"', dash)
            self.assertIn('data-tab="multiyear"', dash)
            self.assertIn('data-tab="methods"', dash)
            self.assertIn('data-tab="sources"', dash)
            self.assertNotIn("Cohort matched", dash)
            self.assertNotIn('data-tab="scenario"', dash)
            self.assertNotIn("Expected vs observed", dash)
            self.assertNotIn("Cross-year trends", dash)
            self.assertNotIn("Global full-text search", dash)
            self.assertNotIn("school-type-global", dash)
            self.assertIn('id="cohort-search"', dash)
            self.assertIn('id="multi-search"', dash)
            self.assertIn('id="cohort-school-type"', dash)
            self.assertIn('id="multi-school-type"', dash)
            self.assertIn('id="cohort-scatter-jpz"', dash)
            self.assertIn('id="cohort-scatter-selectivity"', dash)
            self.assertIn('id="multi-scatter-jpz"', dash)
            self.assertIn('id="multi-scatter-selectivity"', dash)
            self.assertIn("JPZ percentile → MZ percentile", dash)
            self.assertIn("Selectivity percentile (synthetic) → MZ percentile", dash)
            self.assertIn("Expected MZ percentile", dash)
            self.assertIn("Residual (pp)", dash)
            self.assertIn("c==='AJ'?null", dash)
            self.assertIn("Bubble size = MZ throughput proxy (max candidates)", dash)
            # Color mapping without legend markup.
            self.assertIn("schoolTypeColor", dash)
            self.assertIn("SCHOOL_TYPE_PALETTE", dash)
            self.assertNotIn("legend", dash.lower())
            # Expected/residual join uses school_key with redizo fallback, matching cohortRows.
            self.assertIn("schoolKeyOf", dash)
            self.assertIn("e.mz_mean_score_pct_expected", dash)
            self.assertIn("e.residual_pp", dash)
            # MZ score (%) column present alongside MZ percentile.
            self.assertIn("MZ score (%)", dash)
            self.assertIn("r.outcome_mz_mean_score_pct", dash)
            # Multi-year selectivity plot populated (not hardcoded empty array).
            self.assertNotIn("drawScatter('multi-scatter-selectivity',[]", dash)
            self.assertIn("drawScatter('multi-scatter-selectivity',rows,'sel'", dash)
            self.assertIn("selectivityAgg", dash)
            self.assertIn("Avg selectivity percentile (synthetic)", dash)
            # Pagination controls present for both cohort and multi-year tables.
            self.assertIn('id="cohort-page-size"', dash)
            self.assertIn('id="cohort-page-prev"', dash)
            self.assertIn('id="cohort-page-next"', dash)
            self.assertIn('id="cohort-page-info"', dash)
            self.assertIn('id="multi-page-size"', dash)
            self.assertIn('id="multi-page-prev"', dash)
            self.assertIn('id="multi-page-next"', dash)
            self.assertIn('id="multi-page-info"', dash)
            self.assertIn("function paginate(", dash)
            self.assertIn("toFixed(1)", dash)
            self.assertIn("ticks(", dash)
            self.assertIn("Average JPZ percentile across all available JPZ years", dash)
            self.assertIn("Average MZ school mean percentile across all available MZ years", dash)
            self.assertIn("DATA.cross_year?.jpz_rows", dash)
            self.assertIn("DATA.cross_year?.mz_rows", dash)
            self.assertNotIn("Number(null)", dash)
            self.assertIn("addEventListener('change'", dash)
            self.assertNotIn("School key", dash)
            self.assertIn("Address", dash)
            self.assertNotIn("Identity quality", dash)
            self.assertNotIn("Classification quality", dash)
            self.assertNotIn(">City<", dash)
            self.assertNotIn(">Postcode<", dash)
            self.assertNotIn("Programme raw", dash)
            self.assertNotIn("KKOV raw", dash)
            self.assertNotIn("Quality flag", dash)
            self.assertIn("CJ", dash)
            self.assertIn("M", dash)
            self.assertNotIn("https://", dash)
            self.assertNotIn("http://", dash)
            self.assertNotIn("value added", dash.lower())
            script_parser = _DashboardScriptParser()
            script_parser.feed(dash)
            self.assertTrue(script_parser.executable_dashboard_script)
            self.assertIn("const DATA=JSON.parse(document.getElementById('dashboard-data').textContent);", dash)

            # Column sorting: header click handlers wired, sort applied before pagination.
            self.assertIn("function sortRows(", dash)
            self.assertIn("function headerHtml(", dash)
            self.assertIn("function attachSortHandlers(", dash)
            self.assertIn("attachSortHandlers('cohort',el('cohort-head'),renderCohort)", dash)
            self.assertIn("attachSortHandlers('multi',el('multi-head'),renderMulti)", dash)
            sort_cohort_idx = dash.index("rows=sortRows('cohort',rows)")
            paginate_cohort_idx = dash.index("const pageRows=paginate('cohort',rows)")
            self.assertLess(sort_cohort_idx, paginate_cohort_idx)
            sort_multi_idx = dash.index("rows=sortRows('multi',rows)")
            paginate_multi_idx = dash.index("const pageRows=paginate('multi',rows)")
            self.assertLess(sort_multi_idx, paginate_multi_idx)
            # Sort resets to page 1 on click.
            self.assertIn("pageState[prefix]=1;renderFn();", dash)

            # Diagonal reference line only on JPZ->MZ plots, not selectivity plots.
            self.assertIn("drawScatter('cohort-scatter-jpz',rows,'jpz','mz','proxy'", dash)
            self.assertIn(
                "drawScatter('cohort-scatter-jpz',rows,'jpz','mz','proxy','JPZ percentile (0-100)',"
                "'MZ school mean percentile (0-100)','Bubble size = MZ throughput proxy (max candidates)',true)",
                dash,
            )
            self.assertIn(
                "drawScatter('multi-scatter-jpz',rows,'jpz','mz','size','Average JPZ percentile across all available JPZ years',"
                "'Average MZ school mean percentile across all available MZ years',hasProxy?'Bubble size = MZ throughput proxy (max candidates)'"
                ":'Bubble size = cross-year coverage count (max JPZ/MZ years)',true)",
                dash,
            )
            self.assertNotIn(
                "drawScatter('cohort-scatter-selectivity',rows,'sel','mz','proxy','Selectivity percentile (synthetic, 0-100)',"
                "'MZ school mean percentile (0-100)','Bubble size = MZ throughput proxy (max candidates)',true)",
                dash,
            )
            self.assertNotIn(
                "drawScatter('multi-scatter-selectivity',rows,'sel','mz','size','Average selectivity percentile (synthetic, 0-100)',"
                "'Average MZ school mean percentile across all available MZ years',hasProxy?'Bubble size = MZ throughput proxy (max candidates)'"
                ":'Bubble size = cross-year coverage count (max JPZ/MZ years)',true)",
                dash,
            )
            self.assertIn("if(diag){", dash)
            self.assertIn("stroke-dasharray='5,4'", dash)

            jpz = pd.read_csv(Path(result["normalized_dir"]) / "jpz_components.csv")
            comp = jpz[jpz["component"].isin(["CJ", "M", "CJ_M_EQUAL"])].copy()
            one = comp[(comp["entry_year"] == 2018) & (comp["redizo"].astype(str) == "600100001")]
            cj = float(one[one["component"] == "CJ"]["mean_percentile"].iloc[0])
            mm = float(one[one["component"] == "M"]["mean_percentile"].iloc[0])
            eq = float(one[one["component"] == "CJ_M_EQUAL"]["mean_percentile"].iloc[0])
            self.assertAlmostEqual(eq, (cj + mm) / 2.0)
            self.assertEqual(one[one["component"] == "CJ_M_EQUAL"]["metric_name"].iloc[0], "mean_percentile_cj_m_equal_weight")

            cohort = pd.read_csv(Path(result["reports_dir"]) / "cohort_matched" / "cohort_component_panel.csv")
            self.assertIn("mz_school_mean_percentile", cohort.columns)
            self.assertIn("mz_school_mean_percentile_method", cohort.columns)
            self.assertIn("mz_school_mean_percentile_reference", cohort.columns)
            self.assertIn("jpz_registered", cohort.columns)
            self.assertIn("jpz_sat", cohort.columns)
            self.assertTrue(cohort["mz_school_mean_percentile"].notna().any())
            self.assertTrue(cohort["mz_school_mean_percentile_method"].notna().any())
            self.assertTrue(cohort["mz_school_mean_percentile_reference"].notna().any())
            self.assertIn("synthetic_admitted_intake_selectivity_percentile", cohort.columns)
            self.assertIn("synthetic_admitted_intake_selectivity_latent", cohort.columns)
            self.assertIn("jpz_test_takers_cj_m_mean", cohort.columns)
            self.assertIn("capacity_throughput_proxy_candidates", cohort.columns)

            mz_trend = pd.read_csv(Path(result["reports_dir"]) / "cross_year_descriptive" / "mz_school_component_trends.csv")
            self.assertIn("mz_school_mean_percentile", mz_trend.columns)
            self.assertIn("slope_mz_mean_score_pct_per_year", mz_trend.columns)
            self.assertIn("slope_mz_school_mean_percentile_per_year", mz_trend.columns)
            self.assertIn("programme_group_16_raw", mz_trend.columns)
            self.assertIn("kkov_raw", mz_trend.columns)

            unified_csv = pd.read_csv(Path(result["reports_dir"]) / "cross_year_descriptive" / "school_history_unified.csv")
            self.assertIn("slope_mz_mean_score_pct_per_year", unified_csv.columns)
            self.assertIn("slope_mz_school_mean_percentile_per_year", unified_csv.columns)

            model_payload = json.loads((Path(result["reports_dir"]) / "archive_dashboard.html").read_text(encoding="utf-8").split("<script id=\"dashboard-data\" type=\"application/json\">")[1].split("</script>")[0])
            self.assertIn("mz_school_mean_percentile", json.dumps(model_payload, ensure_ascii=False))
            self.assertNotIn("unified_rows", model_payload.get("cross_year", {}))
            self.assertNotIn("cohort", model_payload)
            self.assertIn("scenario_intake", model_payload)
            self.assertIn("cross_year", model_payload)

            school_dim = pd.read_csv(Path(result["normalized_dir"]) / "school_dimension.csv")
            self.assertIn("school_key", school_dim.columns)
            self.assertIn("identity_quality", school_dim.columns)
            self.assertIn("address_raw", school_dim.columns)
            self.assertIn("city", school_dim.columns)
            self.assertIn("school_type", school_dim.columns)
            self.assertIn("classification_quality", school_dim.columns)
            self.assertIn("programme_taxonomy", school_dim.columns)
            self.assertIn("programme_identity", school_dim.columns)
            self.assertTrue(school_dim["address_raw"].notna().any())
            parsed_types = set(school_dim["school_type"].dropna().astype(str).unique().tolist())
            self.assertIn("GY8", parsed_types)
            self.assertIn("GY6", parsed_types)
            self.assertIn("GY4", parsed_types)
            self.assertIn("UNKNOWN", parsed_types)

            payload = json.loads((Path(result["reports_dir"]) / "archive_dashboard.html").read_text(encoding="utf-8").split('<script id="dashboard-data" type="application/json">')[1].split("</script>")[0])
            self.assertIn("coverage", payload)
            self.assertIn("jpz", payload.get("coverage", {}))
            self.assertIn("mz", payload.get("coverage", {}))

            jpz_modern = pd.read_csv(Path(result["normalized_dir"]) / "jpz_modern_round1.csv")
            self.assertIn("programme_group_16_raw", jpz_modern.columns)
            self.assertIn("kkov_raw", jpz_modern.columns)
            self.assertIn("programme_name_raw", jpz_modern.columns)
            self.assertIn("programme_focus_raw", jpz_modern.columns)

            exp = pd.read_csv(Path(result["reports_dir"]) / "cohort_matched" / "expected_vs_observed_association.csv")
            self.assertFalse(exp.empty)
            self.assertIn("residual_pp", exp.columns)
            self.assertIn("mz_mean_score_pct_expected", exp.columns)
            self.assertIn("mz_school_mean_percentile", exp.columns)
            self.assertIn("programme_group_16_raw", exp.columns)
            self.assertIn("kkov_raw", exp.columns)
            self.assertTrue(exp["mz_school_mean_percentile"].notna().any())
            self.assertTrue({"GY8", "GY6", "GY4"}.issubset(set(exp["school_type"].dropna().astype(str).unique().tolist())))
            self.assertNotIn("mz_school_mean_percentile_slope_per_jpz_percentile", exp.columns)
            one_exp = exp.iloc[0]
            lhs = float(one_exp["mz_mean_score_pct_observed"])
            rhs = float(one_exp["mz_mean_score_pct_expected"]) + float(one_exp["residual_pp"])
            self.assertAlmostEqual(lhs, rhs, places=6)

            emeta = json.loads((Path(result["reports_dir"]) / "cohort_matched" / "expected_vs_observed_metadata.json").read_text(encoding="utf-8"))
            self.assertIn("warnings", emeta)
            self.assertIn("mz_school_mean_percentile_slope_per_jpz_percentile", json.dumps(emeta, ensure_ascii=False))
            self.assertIn("defensible 4-year SOS", json.dumps(emeta, ensure_ascii=False))
            self.assertNotIn("value added", json.dumps(emeta, ensure_ascii=False).lower())

            scenario = pd.read_csv(Path(result["reports_dir"]) / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
            self.assertFalse(scenario.empty)
            self.assertIn("synthetic_admitted_intake_selectivity_percentile", scenario.columns)
            self.assertIn("synthetic_admitted_intake_selectivity_latent", scenario.columns)
            self.assertIn("capacity_throughput_proxy_candidates", scenario.columns)
            self.assertIn("outcome_mz_school_mean_percentile", scenario.columns)
            self.assertIn("outcome_score_valid", scenario.columns)
            self.assertIn("outcome_candidates_valid", scenario.columns)
            self.assertIn("scenario_selectivity_valid", scenario.columns)
            self.assertIn("plot_eligible", scenario.columns)
            self.assertIn("plot_exclusion_reason", scenario.columns)
            self.assertIn("outcome_component", scenario.columns)
            self.assertIn("AJ", set(scenario["outcome_component"].dropna().astype(str).unique().tolist()))
            self.assertTrue(scenario["redizo"].astype(str).str.fullmatch(r"\d+").any())
            self.assertTrue(scenario[scenario["outcome_component"] == "AJ"]["synthetic_admitted_intake_selectivity_percentile"].notna().any())

            gy8 = scenario[(scenario["entry_year"] == 2018) & (scenario["school_type"] == "GY8")]
            self.assertFalse(gy8.empty)
            gy8_valid = gy8.dropna(subset=["synthetic_offer_fraction_central", "synthetic_uplift_central", "synthetic_admitted_intake_selectivity_latent", "jpz_test_takers_cj_m_mean", "capacity_throughput_proxy_candidates"])
            if not gy8_valid.empty:
                row = gy8_valid.iloc[0]
                cj_pct = float(row["jpz_cj_mean_percentile"]) / 100.0
                m_pct = float(row["jpz_m_mean_percentile"]) / 100.0
                z_mean = (NormalDist().inv_cdf(max(0.02, min(0.98, cj_pct))) + NormalDist().inv_cdf(max(0.02, min(0.98, m_pct)))) / 2.0
                self.assertNotAlmostEqual(float(row["synthetic_input_cj_m_z_mean"]), (float(row["jpz_cj_mean_percentile"]) + float(row["jpz_m_mean_percentile"])) / 2.0, places=3)
                self.assertAlmostEqual(float(row["synthetic_input_cj_m_z_mean"]), z_mean, places=6)
                p = max(0.02, min(1.0, float(row["capacity_throughput_proxy_candidates"]) / (float(row["jpz_test_takers_cj_m_mean"]) * 0.65)))
                uplift = self._scenario_uplift_expected(p, 0.75)
                self.assertAlmostEqual(float(row["synthetic_offer_fraction_central"]), p, places=6)
                self.assertAlmostEqual(float(row["synthetic_uplift_central"]), uplift, places=6)
                self.assertAlmostEqual(float(row["synthetic_admitted_intake_selectivity_latent"]), z_mean + uplift, places=6)
            else:
                self.assertTrue(gy8["synthetic_admitted_intake_selectivity_latent"].isna().all())

            for (_, st), g in scenario.groupby(["entry_year", "school_type"], dropna=False):
                valid = pd.to_numeric(g["synthetic_admitted_intake_selectivity_percentile"], errors="coerce").dropna()
                if len(valid) > 0:
                    self.assertGreaterEqual(valid.min(), 0.0)
                    self.assertLessEqual(valid.max(), 100.0)

            valid_plot = scenario[scenario["plot_eligible"].fillna(False)]
            self.assertTrue(valid_plot["outcome_mz_school_mean_percentile"].between(0, 100).all())
            self.assertTrue(valid_plot["outcome_mz_mean_score_pct"].notna().all())
            self.assertTrue(valid_plot["capacity_throughput_proxy_candidates"].gt(0).any())
            invalid = scenario[~scenario["plot_eligible"].fillna(False)]
            self.assertTrue(invalid["plot_exclusion_reason"].notna().any())

            pooled = pd.read_csv(Path(result["reports_dir"]) / "cohort_matched" / "pooled_component_association.csv")
            self.assertIn("weighted_slope_mz_school_mean_percentile_per_jpz_percentile", pooled.columns)

    def test_scenario_output_includes_gy6_and_gy8_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            scenario = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
            valid = scenario[scenario["plot_eligible"].fillna(False)]
            self.assertIn("GY6", set(valid["school_type"].dropna().astype(str).unique().tolist()))
            self.assertIn("GY8", set(valid["school_type"].dropna().astype(str).unique().tolist()))
            self.assertTrue(valid["outcome_mz_school_mean_percentile"].between(0, 100).all())

    def test_invalid_scenario_rows_are_excluded_from_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            scenario = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
            invalid = scenario[~scenario["plot_eligible"].fillna(False)]
            self.assertFalse(invalid.empty)
            self.assertTrue(invalid["plot_exclusion_reason"].notna().any())

    def test_historic_multiheader_parser_preserves_registered_and_sat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "JPZ2019_multi.xlsx"
            self._mk_historic_multiheader_fixture(src)
            from gymnazium_value_added.archive_pipeline import parse_jpz_historic_components

            out = parse_jpz_historic_components(src, 2019, "jpz_2019_historic_vysledky")
            self.assertFalse(out.empty)
            c = out[(out["component"] == "CJ") & (out["redizo"].astype(str) == "600199999")].iloc[0]
            m = out[(out["component"] == "M") & (out["redizo"].astype(str) == "600199999")].iloc[0]
            self.assertEqual(int(c["registered"]), 120)
            self.assertEqual(int(c["sat"]), 110)
            self.assertEqual(int(m["registered"]), 118)
            self.assertEqual(int(m["sat"]), 108)
            self.assertTrue(pd.isna(c["admitted"]))

    def test_mz_uses_redizo_smo16_rows_not_redizo_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            mz = pd.read_csv(archive_dir / "normalized" / "mz_components.csv")
            one = mz[(mz["year"] == 2026) & (mz["variant"] == "j") & (mz["redizo"].astype(str) == "600100001") & (mz["school_type"] == "GY8")]
            self.assertFalse(one.empty)
            cj = one[one["component"] == "CJ"].iloc[0]
            total = one[one["component"] == "TOTAL"].iloc[0]
            self.assertEqual(int(cj["candidates"]), 30)
            self.assertAlmostEqual(float(cj["mean_score"]), 74.0)
            self.assertEqual(int(total["candidates"]), 30)
            self.assertAlmostEqual(float(total["mean_score"]), 73.0)

    def test_redizo_join_works_even_with_different_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            # mutate one MZ file school names while keeping same REDIZO
            mzp = files["mz_2026_jap"]
            df = pd.read_excel(mzp)
            df["NÁZEV ŠKOLY"] = ["Gymnázium Město"] * len(df)
            df.to_excel(mzp, index=False)

            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            cohort = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "cohort_component_panel.csv")
            self.assertGreaterEqual(len(cohort), 2)
            self.assertNotIn("school_key", cohort.columns)
            self.assertTrue(cohort["redizo"].astype(str).str.fullmatch(r"\d+").all())

            cross = pd.read_csv(archive_dir / "reports" / "cross_year_descriptive" / "jpz_school_component_trends.csv")
            self.assertIn("n_years", cross.columns)
            self.assertNotIn("graduation_year", cross.columns)

    def test_school_dimension_retains_real_address_and_city(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            school_dim = pd.read_csv(archive_dir / "normalized" / "school_dimension.csv")
            self.assertTrue(school_dim["address_raw"].astype(str).str.contains("Praha|Brno").any())
            self.assertTrue(school_dim["city"].astype(str).str.contains("Praha|Brno").any())
            self.assertTrue(school_dim["canonical_source_id"].notna().all())
            self.assertIn("address_source_id", school_dim.columns)
            self.assertIn("city_source_id", school_dim.columns)

    def test_school_dimension_prefers_best_non_null_source_per_redizo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_rows = pd.DataFrame(
                [
                    {"redizo": "600001", "school_name_raw": "A", "address_raw": pd.NA, "city": pd.NA, "postcode": pd.NA, "source_id": "mz_2025_j"},
                    {"redizo": "600001", "school_name_raw": "A", "address_raw": "Ulice 1, Praha 11000", "city": "Praha", "postcode": "110 00", "source_id": "jpz_2018_historic_vysledky"},
                    {"redizo": "600001", "school_name_raw": "A", "address_raw": "Ulice 1, Praha 11000", "city": "Praha", "postcode": "110 00", "source_id": "mz_2026_j"},
                ]
            )
            from gymnazium_value_added.archive_pipeline import _build_school_dimension

            built = _build_school_dimension(
                pd.DataFrame(columns=["redizo", "school_name_raw", "address_raw", "city", "postcode", "source_id"]),
                pd.DataFrame(columns=["redizo", "school_name_raw", "address_raw", "city", "postcode", "source_id"]),
                source_rows,
            )
            row = built[built["redizo"].astype(str) == "600001"].iloc[0]
            self.assertEqual(row["address_raw"], "Ulice 1, Praha 11000")
            self.assertEqual(row["city"], "Praha")
            self.assertEqual(row["postcode"], "110 00")

    def test_run_requires_explicit_mode_and_supports_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({"mz_2016_j": str(files["mz_2016_j"]), "mz_2016_jap": str(files["mz_2016_jap"])}), encoding="utf-8")
            with self.assertRaises(ValueError):
                main(["run"])
            rc = main([
                "archive-official",
                "--archive-root",
                str(root / "archive"),
                "--year-start",
                "2016",
                "--year-end",
                "2016",
                "--source-override",
                str(ov),
            ])
            self.assertEqual(rc, 0)

    def test_dashboard_chromium_headless_sort_and_diagonal_line(self) -> None:
        chromium = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
        if not chromium:
            self.skipTest("chromium not available")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            dash_path = archive_dir / "reports" / "archive_dashboard.html"
            dash_html = dash_path.read_text(encoding="utf-8")

            probe_script = """
<script>
window.addEventListener('load', () => {
  const before = Array.from(document.querySelectorAll('#cohort-body tr td:nth-child(3)')).map(td => td.textContent);
  document.getElementById('cohort-head').querySelector('th[data-key="redizo"]').click();
  const afterAsc = Array.from(document.querySelectorAll('#cohort-body tr td:nth-child(3)')).map(td => td.textContent);
  document.getElementById('cohort-head').querySelector('th[data-key="redizo"]').click();
  const afterDesc = Array.from(document.querySelectorAll('#cohort-body tr td:nth-child(3)')).map(td => td.textContent);
  const isNonDecreasing = arr => arr.every((v, i) => i === 0 || v >= arr[i - 1]);
  const isNonIncreasing = arr => arr.every((v, i) => i === 0 || v <= arr[i - 1]);
  const jpzDiag = (document.getElementById('cohort-scatter-jpz').innerHTML.match(/stroke-dasharray/g) || []).length;
  const selDiag = (document.getElementById('cohort-scatter-selectivity').innerHTML.match(/stroke-dasharray/g) || []).length;
  const multiJpzDiag = (document.getElementById('multi-scatter-jpz').innerHTML.match(/stroke-dasharray/g) || []).length;
  const multiSelDiag = (document.getElementById('multi-scatter-selectivity').innerHTML.match(/stroke-dasharray/g) || []).length;
  const out = {
    ascDiffersFromUnsorted: JSON.stringify(afterAsc) !== JSON.stringify(before),
    ascIsSortedAscending: isNonDecreasing(afterAsc),
    descIsSortedDescending: isNonIncreasing(afterDesc),
    ascNotEqualDesc: JSON.stringify(afterAsc) !== JSON.stringify(afterDesc),
    jpzDiag, selDiag, multiJpzDiag, multiSelDiag,
  };
  const pre = document.createElement('pre');
  pre.id = 'test-probe-output';
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
});
</script>
</body></html>"""
            probe_html = dash_html.replace("</body></html>", probe_script)
            probe_path = archive_dir / "reports" / "archive_dashboard_probe.html"
            probe_path.write_text(probe_html, encoding="utf-8")

            proc = subprocess.run(
                [
                    chromium,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--dump-dom",
                    "--virtual-time-budget=5000",
                    f"file://{probe_path}",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            dom = proc.stdout
            marker = '<pre id="test-probe-output">'
            self.assertIn(marker, dom, "probe script did not execute; dashboard JS may be broken")
            payload = dom.split(marker, 1)[1].split("</pre>", 1)[0]
            result = json.loads(payload)

            self.assertTrue(result["ascDiffersFromUnsorted"], "clicking header did not change row order")
            self.assertTrue(result["ascIsSortedAscending"], "ascending sort is not in ascending order")
            self.assertTrue(result["descIsSortedDescending"], "second click did not produce descending order")
            self.assertTrue(result["ascNotEqualDesc"], "ascending and descending sort produced identical order")
            self.assertGreater(result["jpzDiag"], 0, "cohort JPZ->MZ plot missing diagonal reference line")
            self.assertGreater(result["multiJpzDiag"], 0, "multi-year JPZ->MZ plot missing diagonal reference line")
            self.assertEqual(result["selDiag"], 0, "cohort selectivity plot should not have diagonal reference line")
            self.assertEqual(result["multiSelDiag"], 0, "multi-year selectivity plot should not have diagonal reference line")

    def test_local_controls_exist_for_cohort_and_multiyear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            dash = (archive_dir / "reports" / "archive_dashboard.html").read_text(encoding="utf-8")
            self.assertIn('id="cohort-grad-year"', dash)
            self.assertNotIn('id="multi-min-years"', dash)
            self.assertIn('id="multi-component"', dash)

    def test_expected_residual_join_populates_cj_m_and_nulls_aj(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)

            def school_key_of(row: dict) -> str:
                sk = row.get("school_key")
                if sk not in (None, ""):
                    return str(sk)
                return f"redizo:{row.get('redizo') or ''}"

            expected = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "expected_vs_observed_association.csv")
            scenario = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
            self.assertFalse(expected.empty)
            self.assertFalse(scenario.empty)

            expected_map = {}
            for _, r in expected.iterrows():
                key = "|".join([school_key_of(r.to_dict()), str(r.get("component") or ""), str(r.get("entry_year") or ""), str(r.get("graduation_year") or "")])
                expected_map[key] = r

            cj_m_found = False
            aj_found = False
            for _, r in scenario.iterrows():
                comp = str(r.get("outcome_component") or "")
                key = "|".join([school_key_of(r.to_dict()), comp, str(r.get("entry_year") or ""), str(r.get("graduation_year") or "")])
                match = expected_map.get(key)
                if comp in ("CJ", "M") and match is not None:
                    cj_m_found = True
                    self.assertTrue(pd.notna(match["mz_mean_score_pct_expected"]))
                    self.assertTrue(pd.notna(match["residual_pp"]))
                if comp == "AJ":
                    aj_found = True
            self.assertTrue(cj_m_found)
            self.assertTrue(aj_found)

    def test_all_type_classification_and_gymnasium_cohort_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)

            mz = pd.read_csv(archive_dir / "normalized" / "mz_components.csv")
            expected_types = {
                "GY8",
                "GY6",
                "GY4",
                "LYC",
                "SOS_TECHNICAL",
                "SOS_ECONOMIC",
                "SOS_HOTEL_BUSINESS",
                "SOS_PEDAGOGICAL_HUMANITIES",
                "SOS_AGRICULTURAL",
                "SOS_HEALTHCARE",
                "SOS_ARTS",
                "SOU_TECHNICAL",
                "SOU_OTHER",
                "NASTAVBA_TECHNICAL",
                "NASTAVBA_OTHER",
                "UNKNOWN",
            }
            self.assertTrue(expected_types.issubset(set(mz["school_type"].dropna().astype(str).unique().tolist())))
            self.assertFalse((mz["school_type"].astype(str) == "TOTAL_AGGREGATE").any())
            self.assertIn("classification_quality", mz.columns)
            self.assertIn("programme_group_16_raw", mz.columns)
            self.assertIn("kkov_raw", mz.columns)
            self.assertIn("programme_name_raw", mz.columns)
            self.assertIn("programme_focus_raw", mz.columns)

            cohort = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "cohort_component_panel.csv")
            self.assertFalse(cohort.empty)
            self.assertEqual(set(cohort["school_type"].dropna().astype(str).unique().tolist()), {"GY8", "GY6", "GY4"})

            payload = json.loads((archive_dir / "reports" / "archive_dashboard.html").read_text(encoding="utf-8").split("<script id=\"dashboard-data\" type=\"application/json\">")[1].split("</script>")[0])
            st_counts = payload.get("coverage", {}).get("school_type_counts", {})
            self.assertIn("GY8", st_counts)
            self.assertIn("GY6", st_counts)
            self.assertIn("GY4", st_counts)
            self.assertIn("UNKNOWN", st_counts)
            self.assertNotIn("TOTAL_AGGREGATE", st_counts)

    def test_sos_matching_is_school_x_smo16_category_four_year_grade9_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            from gymnazium_value_added.archive_pipeline import _cohort_matched

            jpz = pd.DataFrame(
                [
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600500001",
                        "identity_quality": "redizo",
                        "redizo": "600500001",
                        "school_name_raw": "SOS A",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "SOS_TECHNICAL",
                        "programme_taxonomy": "SOS_TECHNICAL",
                        "programme_identity": "SOS_TECHNICAL",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "ST1",
                        "programme_group_16_raw": "ST1",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Strojírenství",
                        "programme_focus_raw": "tech",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "grade": 9,
                        "component": "CJ",
                        "mean_percentile": 52.0,
                        "registered": 100,
                        "sat": 90,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600500001",
                        "identity_quality": "redizo",
                        "redizo": "600500001",
                        "school_name_raw": "SOS A",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "SOS_TECHNICAL",
                        "programme_taxonomy": "SOS_TECHNICAL",
                        "programme_identity": "SOS_TECHNICAL",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "ST1",
                        "programme_group_16_raw": "ST1",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Strojírenství",
                        "programme_focus_raw": "tech",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "grade": 9,
                        "component": "M",
                        "mean_percentile": 48.0,
                        "registered": 100,
                        "sat": 90,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600500002",
                        "identity_quality": "redizo",
                        "redizo": "600500002",
                        "school_name_raw": "SOS B",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": "Brno",
                        "postcode": "602 00",
                        "school_type": "SOS_ECONOMIC",
                        "programme_taxonomy": "SOS_ECONOMIC",
                        "programme_identity": "SOS_ECONOMIC",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "SEK",
                        "programme_group_16_raw": "SEK",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Ekonomika",
                        "programme_focus_raw": "ekon",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "grade": 7,
                        "component": "CJ",
                        "mean_percentile": 44.0,
                        "registered": 80,
                        "sat": 70,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                ]
            )

            mz = pd.DataFrame(
                [
                    {
                        "year": 2022,
                        "variant": "jap",
                        "school_key": "redizo:600500001",
                        "identity_quality": "redizo",
                        "redizo": "600500001",
                        "school_name_raw": "SOS A",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "SOS_TECHNICAL",
                        "programme_taxonomy": "SOS_TECHNICAL",
                        "programme_identity": "SOS_TECHNICAL",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "ST1",
                        "programme_group_16_raw": "ST1",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Strojírenství",
                        "programme_focus_raw": "tech",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "component": "CJ",
                        "candidates": 40,
                        "mean_score": 61.0,
                    },
                    {
                        "year": 2022,
                        "variant": "jap",
                        "school_key": "redizo:600500001",
                        "identity_quality": "redizo",
                        "redizo": "600500001",
                        "school_name_raw": "SOS A",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "SOS_TECHNICAL",
                        "programme_taxonomy": "SOS_TECHNICAL",
                        "programme_identity": "SOS_TECHNICAL",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "ST1",
                        "programme_group_16_raw": "ST1",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Strojírenství",
                        "programme_focus_raw": "tech",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "component": "M",
                        "candidates": 38,
                        "mean_score": 58.0,
                    },
                    {
                        "year": 2022,
                        "variant": "jap",
                        "school_key": "redizo:600500001",
                        "identity_quality": "redizo",
                        "redizo": "600500001",
                        "school_name_raw": "SOS A",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "SOS_ECONOMIC",
                        "programme_taxonomy": "SOS_ECONOMIC",
                        "programme_identity": "SOS_ECONOMIC",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "SEK",
                        "programme_group_16_raw": "SEK",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Ekonomika",
                        "programme_focus_raw": "ekon",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "component": "CJ",
                        "candidates": 50,
                        "mean_score": 70.0,
                    },
                    {
                        "year": 2022,
                        "variant": "jap",
                        "school_key": "redizo:600500002",
                        "identity_quality": "redizo",
                        "redizo": "600500002",
                        "school_name_raw": "SOS B",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": "Brno",
                        "postcode": "602 00",
                        "school_type": "SOS_ECONOMIC",
                        "programme_taxonomy": "SOS_ECONOMIC",
                        "programme_identity": "SOS_ECONOMIC",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "SEK",
                        "programme_group_16_raw": "SEK",
                        "kkov_raw": pd.NA,
                        "programme_name_raw": "Ekonomika",
                        "programme_focus_raw": "ekon",
                        "programme_duration_years": 4,
                        "entrant_grade": 9,
                        "component": "CJ",
                        "candidates": 30,
                        "mean_score": 55.0,
                    },
                ]
            )

            _cohort_matched(jpz, mz, out_dir)
            panel = pd.read_csv(out_dir / "cohort_component_panel.csv")
            self.assertFalse(panel.empty)
            self.assertEqual(set(panel["school_type"].dropna().astype(str).unique().tolist()), {"SOS_TECHNICAL"})
            self.assertEqual(set(panel["component"].dropna().astype(str).unique().tolist()), {"CJ", "M"})
            self.assertTrue((panel["graduation_year"].astype(int) == panel["entry_year"].astype(int) + 4).all())

            meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("school_type(SMO16 taxonomy category)", json.dumps(meta, ensure_ascii=False))
            self.assertIn("sos_match_types", meta)
            self.assertIn("SOS_ARTS", meta.get("sos_match_types", []))

    def test_cohort_matched_keeps_gy6_gy8_when_location_fields_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            from gymnazium_value_added.archive_pipeline import _cohort_matched

            jpz = pd.DataFrame(
                [
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600800001",
                        "identity_quality": "redizo",
                        "redizo": "600800001",
                        "school_name_raw": "Gymnázium GY8",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": pd.NA,
                        "postcode": pd.NA,
                        "school_type": "GY8",
                        "programme_taxonomy": "GY8",
                        "programme_identity": "GY8",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY8",
                        "programme_group_16_raw": "GY8",
                        "kkov_raw": "79-41-K/81",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "8leté",
                        "programme_duration_years": 8,
                        "entrant_grade": 5,
                        "grade": 5,
                        "component": "CJ",
                        "mean_percentile": 65.0,
                        "registered": 100,
                        "sat": 95,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600800001",
                        "identity_quality": "redizo",
                        "redizo": "600800001",
                        "school_name_raw": "Gymnázium GY8",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": pd.NA,
                        "postcode": pd.NA,
                        "school_type": "GY8",
                        "programme_taxonomy": "GY8",
                        "programme_identity": "GY8",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY8",
                        "programme_group_16_raw": "GY8",
                        "kkov_raw": "79-41-K/81",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "8leté",
                        "programme_duration_years": 8,
                        "entrant_grade": 5,
                        "grade": 5,
                        "component": "M",
                        "mean_percentile": 61.0,
                        "registered": 100,
                        "sat": 94,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600800002",
                        "identity_quality": "redizo",
                        "redizo": "600800002",
                        "school_name_raw": "Gymnázium GY6",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": pd.NA,
                        "postcode": pd.NA,
                        "school_type": "GY6",
                        "programme_taxonomy": "GY6",
                        "programme_identity": "GY6",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY6",
                        "programme_group_16_raw": "GY6",
                        "kkov_raw": "79-41-K/61",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "6leté",
                        "programme_duration_years": 6,
                        "entrant_grade": 7,
                        "grade": 7,
                        "component": "CJ",
                        "mean_percentile": 63.0,
                        "registered": 80,
                        "sat": 75,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                    {
                        "entry_year": 2018,
                        "school_key": "redizo:600800002",
                        "identity_quality": "redizo",
                        "redizo": "600800002",
                        "school_name_raw": "Gymnázium GY6",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": pd.NA,
                        "postcode": pd.NA,
                        "school_type": "GY6",
                        "programme_taxonomy": "GY6",
                        "programme_identity": "GY6",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY6",
                        "programme_group_16_raw": "GY6",
                        "kkov_raw": "79-41-K/61",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "6leté",
                        "programme_duration_years": 6,
                        "entrant_grade": 7,
                        "grade": 7,
                        "component": "M",
                        "mean_percentile": 59.0,
                        "registered": 80,
                        "sat": 74,
                        "source_id": "jpz_2018_historic_vysledky",
                    },
                ]
            )

            mz = pd.DataFrame(
                [
                    {
                        "year": 2026,
                        "variant": "jap",
                        "school_key": "redizo:600800001",
                        "identity_quality": "redizo",
                        "redizo": "600800001",
                        "school_name_raw": "Gymnázium GY8",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "GY8",
                        "programme_taxonomy": "GY8",
                        "programme_identity": "GY8",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY8",
                        "programme_group_16_raw": "GY8",
                        "kkov_raw": "79-41-K/81",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "8leté",
                        "programme_duration_years": 8,
                        "entrant_grade": 5,
                        "component": "CJ",
                        "candidates": 30,
                        "mean_score": 80.0,
                    },
                    {
                        "year": 2026,
                        "variant": "jap",
                        "school_key": "redizo:600800001",
                        "identity_quality": "redizo",
                        "redizo": "600800001",
                        "school_name_raw": "Gymnázium GY8",
                        "address_raw": "Ulice 1, Praha 11000",
                        "city": "Praha",
                        "postcode": "110 00",
                        "school_type": "GY8",
                        "programme_taxonomy": "GY8",
                        "programme_identity": "GY8",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY8",
                        "programme_group_16_raw": "GY8",
                        "kkov_raw": "79-41-K/81",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "8leté",
                        "programme_duration_years": 8,
                        "entrant_grade": 5,
                        "component": "M",
                        "candidates": 29,
                        "mean_score": 78.0,
                    },
                    {
                        "year": 2024,
                        "variant": "jap",
                        "school_key": "redizo:600800002",
                        "identity_quality": "redizo",
                        "redizo": "600800002",
                        "school_name_raw": "Gymnázium GY6",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": "Brno",
                        "postcode": "602 00",
                        "school_type": "GY6",
                        "programme_taxonomy": "GY6",
                        "programme_identity": "GY6",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY6",
                        "programme_group_16_raw": "GY6",
                        "kkov_raw": "79-41-K/61",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "6leté",
                        "programme_duration_years": 6,
                        "entrant_grade": 7,
                        "component": "CJ",
                        "candidates": 25,
                        "mean_score": 79.0,
                    },
                    {
                        "year": 2024,
                        "variant": "jap",
                        "school_key": "redizo:600800002",
                        "identity_quality": "redizo",
                        "redizo": "600800002",
                        "school_name_raw": "Gymnázium GY6",
                        "address_raw": "Ulice 2, Brno 60200",
                        "city": "Brno",
                        "postcode": "602 00",
                        "school_type": "GY6",
                        "programme_taxonomy": "GY6",
                        "programme_identity": "GY6",
                        "classification_quality": "authoritative_programme_group16",
                        "programme_group_raw": "GY6",
                        "programme_group_16_raw": "GY6",
                        "kkov_raw": "79-41-K/61",
                        "programme_name_raw": "Gymnázium",
                        "programme_focus_raw": "6leté",
                        "programme_duration_years": 6,
                        "entrant_grade": 7,
                        "component": "M",
                        "candidates": 24,
                        "mean_score": 77.0,
                    },
                ]
            )

            _cohort_matched(jpz, mz, out_dir)
            scenario = pd.read_csv(out_dir / "scenario_intake_vs_mz_outcomes.csv")
            valid = scenario[scenario["plot_eligible"].fillna(False)]
            self.assertIn("GY8", set(valid["school_type"].dropna().astype(str).unique().tolist()))
            self.assertIn("GY6", set(valid["school_type"].dropna().astype(str).unique().tolist()))

    def test_sos_smo16_taxonomy_carries_authoritative_matching_fields(self) -> None:
        from gymnazium_value_added.archive_pipeline import _attach_school_type_fields

        df = pd.DataFrame(
            [
                {"programme_group_16_raw": "ST1", "grade": pd.NA},
                {"programme_group_16_raw": "SEK", "grade": pd.NA},
                {"programme_group_16_raw": "SUM", "grade": pd.NA},
            ]
        )
        out = _attach_school_type_fields(df)

        self.assertEqual(out["programme_duration_years"].dropna().astype(int).tolist(), [4, 4, 4])
        self.assertEqual(out["entrant_grade"].dropna().astype(int).tolist(), [9, 9, 9])
        self.assertEqual(out["classification_quality"].dropna().astype(str).unique().tolist(), ["authoritative_programme_group16"])

    def test_synthetic_selectivity_null_without_capacity_proxy(self) -> None:
        from gymnazium_value_added.archive_pipeline import _build_scenario_selectivity_and_outcomes

        jpz = pd.DataFrame(
            [
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 60.0,
                    "sat": 100,
                    "registered": 110,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 55.0,
                    "sat": 90,
                    "registered": 95,
                },
            ]
        )
        mz_pref = pd.DataFrame(
            [
                {
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "year": 2026,
                    "component": "AJ",
                    "mean_score": 72.0,
                    "candidates": pd.NA,
                    "variant": "jap",
                    "source_id": "mz_2026_jap",
                }
            ]
        )
        out = _build_scenario_selectivity_and_outcomes(jpz, mz_pref)
        self.assertFalse(out.empty)
        self.assertTrue(pd.isna(out.iloc[0]["capacity_throughput_proxy_candidates"]))
        self.assertTrue(pd.isna(out.iloc[0]["synthetic_admitted_intake_selectivity_latent"]))
        self.assertTrue(pd.isna(out.iloc[0]["synthetic_admitted_intake_selectivity_percentile"]))

    def test_synthetic_selectivity_uses_capacity_proxy_when_historic_sat_and_registered_missing(self) -> None:
        from gymnazium_value_added.archive_pipeline import _build_scenario_selectivity_and_outcomes

        jpz = pd.DataFrame(
            [
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 60.0,
                    "sat": pd.NA,
                    "registered": pd.NA,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 55.0,
                    "sat": pd.NA,
                    "registered": pd.NA,
                },
            ]
        )
        mz_pref = pd.DataFrame(
            [
                {
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "year": 2026,
                    "component": "CJ",
                    "mean_score": 72.0,
                    "candidates": 60,
                    "variant": "jap",
                    "source_id": "mz_2026_jap",
                },
                {
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "year": 2026,
                    "component": "M",
                    "mean_score": 71.0,
                    "candidates": 60,
                    "variant": "jap",
                    "source_id": "mz_2026_jap",
                },
                {
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "year": 2026,
                    "component": "AJ",
                    "mean_score": 73.0,
                    "candidates": 60,
                    "variant": "jap",
                    "source_id": "mz_2026_jap",
                },
            ]
        )
        out = _build_scenario_selectivity_and_outcomes(jpz, mz_pref)
        self.assertFalse(out.empty)
        row = out.iloc[0]
        self.assertTrue(pd.notna(row["capacity_throughput_proxy_candidates"]))
        self.assertTrue(pd.notna(row["jpz_test_takers_cj_m_mean"]))
        self.assertAlmostEqual(float(row["jpz_test_takers_cj_m_mean"]), 60.0 / 0.65, places=6)
        self.assertTrue(pd.notna(row["synthetic_admitted_intake_selectivity_latent"]))
        self.assertTrue(pd.notna(row["synthetic_admitted_intake_selectivity_percentile"]))

    def test_synthetic_selectivity_gy6_observed_and_gy8_proxy_denominator_label(self) -> None:
        from gymnazium_value_added.archive_pipeline import _build_scenario_selectivity_and_outcomes

        jpz = pd.DataFrame(
            [
                {
                    "entry_year": 2020,
                    "programme_duration_years": 6,
                    "school_key": "redizo:10",
                    "match_school_type": "GY6",
                    "redizo": "10",
                    "school_name_raw": "GY6 A",
                    "address_raw": "U 10",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY6",
                    "programme_identity": "GY6",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 62.0,
                    "sat": 80,
                    "registered": 85,
                },
                {
                    "entry_year": 2020,
                    "programme_duration_years": 6,
                    "school_key": "redizo:10",
                    "match_school_type": "GY6",
                    "redizo": "10",
                    "school_name_raw": "GY6 A",
                    "address_raw": "U 10",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY6",
                    "programme_identity": "GY6",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 58.0,
                    "sat": 78,
                    "registered": 83,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:11",
                    "match_school_type": "GY8",
                    "redizo": "11",
                    "school_name_raw": "GY8 B",
                    "address_raw": "U 11",
                    "city": "Brno",
                    "postcode": "602 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 57.0,
                    "sat": pd.NA,
                    "registered": pd.NA,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:11",
                    "match_school_type": "GY8",
                    "redizo": "11",
                    "school_name_raw": "GY8 B",
                    "address_raw": "U 11",
                    "city": "Brno",
                    "postcode": "602 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 56.0,
                    "sat": pd.NA,
                    "registered": pd.NA,
                },
            ]
        )
        mz_pref = pd.DataFrame(
            [
                {"school_key": "redizo:10", "match_school_type": "GY6", "year": 2026, "component": "CJ", "mean_score": 71.0, "candidates": 52, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:10", "match_school_type": "GY6", "year": 2026, "component": "M", "mean_score": 70.0, "candidates": 52, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:10", "match_school_type": "GY6", "year": 2026, "component": "AJ", "mean_score": 72.0, "candidates": 52, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:11", "match_school_type": "GY8", "year": 2026, "component": "CJ", "mean_score": 68.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:11", "match_school_type": "GY8", "year": 2026, "component": "M", "mean_score": 67.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:11", "match_school_type": "GY8", "year": 2026, "component": "AJ", "mean_score": 69.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:12", "match_school_type": "GY8", "year": 2026, "component": "AJ", "mean_score": 65.0, "candidates": pd.NA, "variant": "jap", "source_id": "mz_2026_jap"},
            ]
        )

        out = _build_scenario_selectivity_and_outcomes(jpz, mz_pref)
        self.assertFalse(out.empty)

        gy6_row = out[out["school_key"].eq("redizo:10")].iloc[0]
        self.assertAlmostEqual(float(gy6_row["jpz_test_takers_cj_m_mean"]), (80.0 + 78.0) / 2.0, places=6)
        self.assertEqual(str(gy6_row["jpz_test_takers_cj_m_mean_method"]), "jpz_sat_or_registered_mean")
        self.assertEqual(str(gy6_row["synthetic_offer_fraction_denominator_source"]), "observed_jpz_test_takers_cj_m_mean")
        self.assertTrue(pd.notna(gy6_row["synthetic_admitted_intake_selectivity_percentile"]))

        gy8_proxy_row = out[out["school_key"].eq("redizo:11")].iloc[0]
        self.assertAlmostEqual(float(gy8_proxy_row["jpz_test_takers_cj_m_mean"]), 60.0 / 0.65, places=6)
        self.assertEqual(str(gy8_proxy_row["jpz_test_takers_cj_m_mean_method"]), "proxy_div_yield_central")
        self.assertEqual(
            str(gy8_proxy_row["synthetic_offer_fraction_denominator_source"]),
            "proxy_derived_from_capacity_proxy_div_yield_central",
        )
        self.assertTrue(pd.notna(gy8_proxy_row["synthetic_admitted_intake_selectivity_percentile"]))

        self.assertTrue(out[out["school_key"].eq("redizo:12")]["synthetic_admitted_intake_selectivity_latent"].isna().all())

    def test_synthetic_selectivity_percentile_is_within_entry_year_school_type_group(self) -> None:
        from gymnazium_value_added.archive_pipeline import _build_scenario_selectivity_and_outcomes

        jpz = pd.DataFrame(
            [
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 70.0,
                    "sat": 100,
                    "registered": 120,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:1",
                    "match_school_type": "GY8",
                    "redizo": "1",
                    "school_name_raw": "A",
                    "address_raw": "U 1",
                    "city": "Praha",
                    "postcode": "110 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 68.0,
                    "sat": 100,
                    "registered": 120,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:2",
                    "match_school_type": "GY8",
                    "redizo": "2",
                    "school_name_raw": "B",
                    "address_raw": "U 2",
                    "city": "Brno",
                    "postcode": "602 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "CJ",
                    "mean_percentile": 45.0,
                    "sat": 100,
                    "registered": 120,
                },
                {
                    "entry_year": 2018,
                    "programme_duration_years": 8,
                    "school_key": "redizo:2",
                    "match_school_type": "GY8",
                    "redizo": "2",
                    "school_name_raw": "B",
                    "address_raw": "U 2",
                    "city": "Brno",
                    "postcode": "602 00",
                    "programme_taxonomy": "GY8",
                    "programme_identity": "GY8",
                    "classification_quality": "ok",
                    "component": "M",
                    "mean_percentile": 43.0,
                    "sat": 100,
                    "registered": 120,
                },
            ]
        )
        mz_pref = pd.DataFrame(
            [
                {"school_key": "redizo:1", "match_school_type": "GY8", "year": 2026, "component": "CJ", "mean_score": 80.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:1", "match_school_type": "GY8", "year": 2026, "component": "M", "mean_score": 79.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:1", "match_school_type": "GY8", "year": 2026, "component": "AJ", "mean_score": 81.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:2", "match_school_type": "GY8", "year": 2026, "component": "CJ", "mean_score": 60.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:2", "match_school_type": "GY8", "year": 2026, "component": "M", "mean_score": 59.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
                {"school_key": "redizo:2", "match_school_type": "GY8", "year": 2026, "component": "AJ", "mean_score": 61.0, "candidates": 60, "variant": "jap", "source_id": "mz_2026_jap"},
            ]
        )
        out = _build_scenario_selectivity_and_outcomes(jpz, mz_pref)
        self.assertFalse(out.empty)
        unique_by_school = out.drop_duplicates(subset=["school_key"])
        a = unique_by_school[unique_by_school["school_key"] == "redizo:1"].iloc[0]
        b = unique_by_school[unique_by_school["school_key"] == "redizo:2"].iloc[0]
        self.assertGreater(float(a["synthetic_admitted_intake_selectivity_latent"]), float(b["synthetic_admitted_intake_selectivity_latent"]))
        self.assertGreater(float(a["synthetic_admitted_intake_selectivity_percentile"]), float(b["synthetic_admitted_intake_selectivity_percentile"]))
        self.assertEqual(sorted(unique_by_school["synthetic_admitted_intake_selectivity_percentile"].dropna().tolist()), [0.0, 100.0])

    def test_analyze_local_fails_when_required_local_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            mpath = archive_dir / "manifest.json"
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
            for src in manifest["sources"]:
                if src.get("source_id") == "jpz_2024_kolo1_prihlasky":
                    src["status"] = "unavailable"
                    src["required"] = True
                    src["reason"] = "fixture-delete"
                    break
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                analyze_archive_local(archive_dir)

    def test_analyze_local_reports_malformed_mz_as_error_with_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            broken_j = files["mz_2026_j"]
            broken_jap = files["mz_2026_jap"]
            broken_df = pd.DataFrame(
                {
                    "REDIZO": ["600100001"],
                    "NÁZEV ŠKOLY": ["Gymnázium A"],
                    "SMO16": ["GY8"],
                    "ČJ KONALI": [30],
                    "ČJ PRŮMĚRNÝ % SKÓR": [70.0],
                }
            )
            broken_df.to_excel(broken_j, index=False)
            broken_df.to_excel(broken_jap, index=False)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov, year_start=2026, year_end=2026)
            with self.assertRaises(ValueError):
                analyze_archive_local(archive_dir)
            diag = archive_dir / "reports" / "parser_diagnostics.json"
            self.assertTrue(diag.exists())

    def test_mz_participation_rate_unit_cj_is_one_m_aj_lower_null_safe(self) -> None:
        from gymnazium_value_added.archive_pipeline import _compute_mz_participation_rate_vs_cj

        mz_pref = pd.DataFrame(
            [
                {"school_key": "redizo:600100001", "match_school_type": "GY8", "year": 2026, "component": "CJ", "candidates": 30},
                {"school_key": "redizo:600100001", "match_school_type": "GY8", "year": 2026, "component": "M", "candidates": 18},
                {"school_key": "redizo:600100001", "match_school_type": "GY8", "year": 2026, "component": "AJ", "candidates": 9},
                # multiple programme rows sharing the same key -> candidates should be summed
                {"school_key": "redizo:600100002", "match_school_type": "GY6", "year": 2026, "component": "CJ", "candidates": 10},
                {"school_key": "redizo:600100002", "match_school_type": "GY6", "year": 2026, "component": "CJ", "candidates": 5},
                {"school_key": "redizo:600100002", "match_school_type": "GY6", "year": 2026, "component": "M", "candidates": 6},
                # missing CJ -> null rate
                {"school_key": "redizo:600100003", "match_school_type": "GY4", "year": 2026, "component": "M", "candidates": 4},
                # zero CJ candidates -> null rate (guard against division by zero)
                {"school_key": "redizo:600100004", "match_school_type": "GY4", "year": 2026, "component": "CJ", "candidates": 0},
                {"school_key": "redizo:600100004", "match_school_type": "GY4", "year": 2026, "component": "M", "candidates": 3},
            ]
        )
        out = _compute_mz_participation_rate_vs_cj(mz_pref)

        def rate(school_key: str, component: str) -> float | None:
            row = out[(out["school_key"] == school_key) & (out["component"] == component)]
            self.assertFalse(row.empty)
            val = row.iloc[0]["mz_participation_rate_vs_cj"]
            return None if pd.isna(val) else float(val)

        self.assertAlmostEqual(rate("redizo:600100001", "CJ"), 1.0, places=6)
        self.assertAlmostEqual(rate("redizo:600100001", "M"), 18 / 30, places=6)
        self.assertAlmostEqual(rate("redizo:600100001", "AJ"), 9 / 30, places=6)
        self.assertLess(rate("redizo:600100001", "M"), rate("redizo:600100001", "CJ"))
        self.assertLess(rate("redizo:600100001", "AJ"), rate("redizo:600100001", "CJ"))

        # summed candidates across duplicate programme rows sharing the same key
        self.assertAlmostEqual(rate("redizo:600100002", "CJ"), 1.0, places=6)
        self.assertAlmostEqual(rate("redizo:600100002", "M"), 6 / 15, places=6)

        # missing CJ / zero CJ -> null (division guard)
        self.assertIsNone(rate("redizo:600100003", "M"))
        self.assertIsNone(rate("redizo:600100004", "CJ"))
        self.assertIsNone(rate("redizo:600100004", "M"))

    def test_mz_participation_rate_empty_input_returns_empty_frame_with_expected_columns(self) -> None:
        from gymnazium_value_added.archive_pipeline import _compute_mz_participation_rate_vs_cj

        out = _compute_mz_participation_rate_vs_cj(pd.DataFrame())
        self.assertTrue(out.empty)
        for col in ["school_key", "match_school_type", "year", "component", "mz_participation_rate_vs_cj"]:
            self.assertIn(col, out.columns)

    def test_mz_participation_rate_exposed_in_cohort_and_scenario_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)

            panel = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "cohort_component_panel.csv")
            self.assertIn("mz_participation_rate_vs_cj", panel.columns)
            cj_panel = panel[panel["component"] == "CJ"]["mz_participation_rate_vs_cj"].dropna()
            self.assertFalse(cj_panel.empty)
            self.assertTrue((cj_panel.between(0.99, 1.01)).all())

            scenario = pd.read_csv(archive_dir / "reports" / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
            self.assertIn("mz_participation_rate_vs_cj", scenario.columns)
            cj_scn = scenario[scenario["outcome_component"] == "CJ"]["mz_participation_rate_vs_cj"].dropna()
            self.assertFalse(cj_scn.empty)
            self.assertTrue((cj_scn.between(0.99, 1.01)).all())
            # AJ rows are present with a participation rate in scenario outcomes (panel only carries CJ/M).
            aj_scn = scenario[scenario["outcome_component"] == "AJ"]["mz_participation_rate_vs_cj"].dropna()
            self.assertFalse(aj_scn.empty)

    def test_dashboard_has_mz_participation_rate_columns_and_methods_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            dashboard = write_archive_dashboard(archive_dir)
            html_text = dashboard.read_text(encoding="utf-8")

            self.assertIn("MZ participation rate (relative to CJ, %)", html_text)
            self.assertIn("Avg MZ participation rate (relative to CJ, %)", html_text)
            self.assertIn("compulsory-cohort", html_text)
            self.assertIn("repeat", html_text)

    def test_optimized_dashboard_payload_columns_and_absent_unused_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self._mk_sources(root)
            ov = root / "override.json"
            ov.write_text(json.dumps({k: str(v) for k, v in files.items()}), encoding="utf-8")
            archive_dir = create_archive(root / "archive", source_override=ov)
            analyze_archive_local(archive_dir)
            dashboard = write_archive_dashboard(archive_dir)
            html_text = dashboard.read_text(encoding="utf-8")

            # 1. Assert absent/empty unused datasets
            payload = json.loads(html_text.split('<script id="dashboard-data" type="application/json">')[1].split("</script>")[0])
            self.assertNotIn("cohort", payload)
            self.assertNotIn("unified_rows", payload.get("cross_year", {}))

            # 2. Assert projected datasets have all fields JS accesses
            scenario_cols = set(payload.get("scenario_intake", {}).get("rows", {}).get("cols", []))
            for req in [
                "school_key", "redizo", "school_type", "outcome_component",
                "entry_year", "graduation_year", "jpz_mean_percentile",
                "synthetic_admitted_intake_selectivity_percentile",
                "outcome_mz_school_mean_percentile", "outcome_mz_mean_score_pct",
                "capacity_throughput_proxy_candidates", "mz_participation_rate_vs_cj",
            ]:
                self.assertIn(req, scenario_cols)
            self.assertTrue(bool({"school_name_raw", "school_name_raw_jpz"}.intersection(scenario_cols)))
            self.assertTrue(bool({"address_raw", "address_raw_jpz"}.intersection(scenario_cols)))

            expected_cols = set(payload.get("expected_observed", {}).get("rows", {}).get("cols", []))
            for req in [
                "redizo", "component", "entry_year", "graduation_year",
                "mz_mean_score_pct_expected", "residual_pp",
            ]:
                self.assertIn(req, expected_cols)

            jpz_cols = set(payload.get("cross_year", {}).get("jpz_rows", {}).get("cols", []))
            for req in ["school_key", "redizo", "school_name_raw", "address_raw", "school_type", "component", "mean_jpz_percentile", "n_years"]:
                self.assertIn(req, jpz_cols)

            mz_cols = set(payload.get("cross_year", {}).get("mz_rows", {}).get("cols", []))
            for req in ["school_key", "redizo", "school_name_raw", "address_raw", "school_type", "component", "mz_school_mean_percentile", "n_years", "max_mz_candidates", "mean_mz_participation_rate_vs_cj"]:
                self.assertIn(req, mz_cols)

            # 3. Assert numeric float rounding
            def check_rounding(node: object) -> None:
                if isinstance(node, float):
                    s = str(node)
                    if "." in s:
                        dec = len(s.split(".")[1])
                        self.assertLessEqual(dec, 2, f"Float {node} has more than 2 decimal places")
                elif isinstance(node, list):
                    for x in node:
                        check_rounding(x)
                elif isinstance(node, dict):
                    for v in node.values():
                        check_rounding(v)

            check_rounding(payload.get("scenario_intake", {}).get("rows"))
            check_rounding(payload.get("expected_observed", {}).get("rows"))
            check_rounding(payload.get("cross_year", {}).get("jpz_rows"))
            check_rounding(payload.get("cross_year", {}).get("mz_rows"))

    def test_target_archive_dashboard_file_size_under_limit(self) -> None:
        target_archive = Path("data/archive/20260807T144301Z")
        if not (target_archive / "reports" / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv").exists():
            self.skipTest("target archive reports not present")

        out_path = write_archive_dashboard(target_archive)
        size_bytes = out_path.stat().st_size
        self.assertLess(size_bytes, 30_000_000, f"Dashboard size {size_bytes} exceeds 30MB limit")
        self.assertLess(size_bytes, 15_000_000, f"Dashboard size {size_bytes} exceeds 15MB target")


if __name__ == "__main__":
    unittest.main()
