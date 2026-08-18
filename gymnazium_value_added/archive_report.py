from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

_URL_RE = re.compile(r"https?://\S+")


def _safe_json(obj: Any) -> str:
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records", force_ascii=False))


def _to_columnar(df: pd.DataFrame, keep_cols: list[str]) -> dict[str, Any]:
    if df.empty:
        return {"cols": [c for c in keep_cols if c in df.columns], "rows": []}
    cols = [c for c in keep_cols if c in df.columns]
    sub = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].round(2)
    records = json.loads(sub.where(pd.notna(sub), None).to_json(orient="values", force_ascii=False))
    return {"cols": cols, "rows": records}


def _dashboard_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    hidden = {
        c
        for c in df.columns
        if c == "quality_flag"
        or c.startswith("kkov")
        or c.startswith("programme_")
        or c.startswith("program_")
    }
    return _to_records(df.drop(columns=hidden, errors="ignore"))


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if str(value).strip():
            return value
    return None


def _redizo_from_school_key(value: Any) -> str | None:
    text = str(value or "").strip()
    m = re.match(r"^redizo:(\d{6,10})$", text)
    return m.group(1) if m else None


def _label_component(value: Any) -> str:
    text = str(value or "").strip()
    return {"CJ": "CJ", "M": "M", "AJ": "AJ", "TOTAL": "TOTAL", "CJ_M_EQUAL": "CJ_M_EQUAL"}.get(text, text)


def _equal_weight_percentile_rank(values: pd.Series) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna()
    n = int(mask.sum())
    out = pd.Series(pd.NA, index=numeric.index, dtype="Float64")
    if n == 0:
        return out, 0
    if n == 1:
        out.loc[mask] = 50.0
        return out, 1
    ranks = numeric.rank(method="average", ascending=True)
    out.loc[mask] = ((ranks.loc[mask] - 1.0) / (n - 1.0) * 100.0).astype(float)
    return out, n


def _linear_slope(years: pd.Series, values: pd.Series) -> float | None:
    y = pd.to_numeric(values, errors="coerce")
    x = pd.to_numeric(years, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return None
    return float(np.polyfit(x[mask].astype(float).to_numpy(), y[mask].astype(float).to_numpy(), 1)[0])


def _strip_urls(value: Any) -> Any:
    if isinstance(value, str):
        return _URL_RE.sub("[URL removed]", value)
    if isinstance(value, list):
        return [_strip_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_urls(item) for key, item in value.items()}
    return value


def _build_unified_school_history(
    school_dim: pd.DataFrame,
    jpz_components: pd.DataFrame,
    mz_components: pd.DataFrame,
    jpz_modern: pd.DataFrame,
) -> pd.DataFrame:
    dim = school_dim.copy() if not school_dim.empty else pd.DataFrame()
    for df, id_col in ((jpz_components, "entry_year"), (mz_components, "year"), (jpz_modern, "entry_year")):
        if df.empty or "school_key" not in df.columns:
            continue
        base_cols = [c for c in ["school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode", id_col, "source_id"] if c in df.columns]
        cur = df[base_cols].copy()
        if dim.empty:
            dim = cur.rename(columns={id_col: "year_hint"})
            continue
        needed = ["school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode"]
        dtmp = dim[[c for c in needed if c in dim.columns]].copy()
        dim = pd.concat([dtmp, cur[[c for c in needed if c in cur.columns]].copy()], ignore_index=True).drop_duplicates(subset=["school_key"], keep="first").reset_index(drop=True)

    if dim.empty:
        dim = pd.DataFrame(columns=["school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode"])

    rows: list[dict[str, Any]] = []
    if not jpz_components.empty:
        jh = jpz_components[jpz_components["component"].isin(["CJ", "M"])].copy()
        for _, r in jh.iterrows():
            rows.append({
                "school_key": r.get("school_key"), "component": _label_component(r.get("component")), "entry_year": r.get("entry_year"), "graduation_year": None,
                "jpz_published_mean_percentile": r.get("mean_percentile"), "jpz_percentile_unit": "percentile", "jpz_registered": r.get("registered"), "jpz_sat": r.get("sat"), "jpz_admitted": r.get("admitted"), "jpz_metric_family": "historic_published_test_result_aggregate",
                "mz_mean_score_pct": None, "mz_school_mean_percentile": None, "mz_school_mean_percentile_method": None, "mz_school_mean_percentile_reference": None,
                "slope_mz_mean_score_pct_per_year": None, "slope_mz_school_mean_percentile_per_year": None, "mz_school_mean_percentile_n_schools": None, "mz_candidates": None, "mz_pass_rate": None,
                "match_status": "jpz_only", "source": r.get("source_id"),
            })

    mz_pref = pd.DataFrame()
    mz_slope_map: dict[tuple[object, object], dict[str, Any]] = {}
    if not mz_components.empty:
        mz_pref = (
            mz_components.assign(_rank=lambda d: d.get("variant", pd.Series([None] * len(d))).map({"jap": 0, "j": 1}).fillna(9))
            .sort_values(["year", "school_key", "component", "_rank"]).drop_duplicates(["year", "school_key", "component"], keep="first")
        )
        mz_pref["mz_school_mean_percentile"] = pd.NA
        mz_pref["mz_school_mean_percentile_n_schools"] = pd.NA
        for (_, _), g in mz_pref.groupby(["year", "component"], dropna=False):
            pct, n = _equal_weight_percentile_rank(g["mean_score"])
            mz_pref.loc[g.index, "mz_school_mean_percentile"] = pct
            mz_pref.loc[g.index, "mz_school_mean_percentile_n_schools"] = n
        mz_pref["mz_school_mean_percentile_method"] = "equal_weight_school_mean_percentile_rank"
        mz_pref["mz_school_mean_percentile_reference"] = mz_pref.apply(lambda r: f"eligible schools in graduation_year={r['year']} component={r['component']}", axis=1)
        mz_pref["slope_mz_mean_score_pct_per_year"] = pd.NA
        mz_pref["slope_mz_school_mean_percentile_per_year"] = pd.NA
        for (school_key, component), g in mz_pref.groupby(["school_key", "component"], dropna=False):
            mz_pref.loc[g.index, "slope_mz_mean_score_pct_per_year"] = _linear_slope(g["year"], g["mean_score"])
            mz_pref.loc[g.index, "slope_mz_school_mean_percentile_per_year"] = _linear_slope(g["year"], g["mz_school_mean_percentile"])
            mz_slope_map[(school_key, component)] = {
                "slope_mz_mean_score_pct_per_year": mz_pref.loc[g.index, "slope_mz_mean_score_pct_per_year"].iloc[0],
                "slope_mz_school_mean_percentile_per_year": mz_pref.loc[g.index, "slope_mz_school_mean_percentile_per_year"].iloc[0],
            }
        for _, r in mz_pref.iterrows():
            rows.append({
                "school_key": r.get("school_key"), "component": _label_component(r.get("component")), "entry_year": None, "graduation_year": r.get("year"),
                "jpz_published_mean_percentile": None, "jpz_percentile_unit": "percentile", "jpz_registered": None, "jpz_sat": None, "jpz_admitted": None, "jpz_metric_family": "historic_published_test_result_aggregate",
                "mz_mean_score_pct": r.get("mean_score"), "mz_school_mean_percentile": r.get("mz_school_mean_percentile"), "mz_school_mean_percentile_method": r.get("mz_school_mean_percentile_method"), "mz_school_mean_percentile_reference": r.get("mz_school_mean_percentile_reference"),
                "slope_mz_mean_score_pct_per_year": r.get("slope_mz_mean_score_pct_per_year"), "slope_mz_school_mean_percentile_per_year": r.get("slope_mz_school_mean_percentile_per_year"), "mz_school_mean_percentile_n_schools": r.get("mz_school_mean_percentile_n_schools"),
                "mz_candidates": r.get("candidates"), "mz_pass_rate": r.get("pass_rate"), "match_status": "mz_only", "source": r.get("source_id"),
            })

    if not jpz_modern.empty:
        modern = jpz_modern.copy()
        modern["component"] = modern.get("actual_score_field", pd.Series([None] * len(modern))).map(lambda x: "M" if isinstance(x, str) and "MA" in x.upper() else ("CJ" if isinstance(x, str) and "CJ" in x.upper() else ""))
        for _, r in modern.iterrows():
            md = pd.to_numeric(pd.Series([r.get("programme_duration_years")]), errors="coerce").iloc[0]
            me = pd.to_numeric(pd.Series([r.get("entry_year")]), errors="coerce").iloc[0]
            rows.append({
                "school_key": r.get("school_key"), "component": _label_component(r.get("component")) or "MODERN", "entry_year": r.get("entry_year"), "graduation_year": me + md if pd.notna(me) and pd.notna(md) else None,
                "jpz_published_mean_percentile": None, "jpz_percentile_unit": None, "jpz_registered": r.get("applications"), "jpz_sat": None, "jpz_admitted": None, "jpz_metric_family": "modern_round1_triplet_separate_family",
                "mz_mean_score_pct": None, "mz_school_mean_percentile": None, "mz_school_mean_percentile_method": None, "mz_school_mean_percentile_reference": None, "slope_mz_mean_score_pct_per_year": None, "slope_mz_school_mean_percentile_per_year": None, "mz_school_mean_percentile_n_schools": None,
                "mz_candidates": None, "mz_pass_rate": None, "match_status": "jpz_modern_only", "source": r.get("source_id"),
                "modern_actual_score_value": r.get("actual_score_value"), "modern_actual_score_unit": r.get("actual_score_unit"), "modern_capacity": r.get("capacity"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode", "component", "entry_year", "graduation_year",
            "jpz_published_mean_percentile", "jpz_percentile_unit", "jpz_registered", "jpz_sat", "jpz_admitted", "jpz_metric_family",
            "mz_mean_score_pct", "mz_school_mean_percentile", "mz_school_mean_percentile_method", "mz_school_mean_percentile_reference",
            "slope_mz_mean_score_pct_per_year", "slope_mz_school_mean_percentile_per_year", "mz_school_mean_percentile_n_schools", "mz_candidates", "mz_pass_rate",
            "match_status", "source", "modern_actual_score_value", "modern_actual_score_unit", "modern_capacity",
        ])
    out = out.merge(dim, on="school_key", how="left", suffixes=("", "_dim"))
    for col in ["identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode"]:
        dim_col = f"{col}_dim"
        if col in out.columns and dim_col in out.columns:
            out[col] = out[col].where(out[col].notna(), out[dim_col])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_dim")], errors="ignore")
    out["redizo"] = out["redizo"].where(out["redizo"].notna(), out["school_key"].map(_redizo_from_school_key))
    keep = [
        "school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode", "component", "entry_year", "graduation_year",
        "jpz_published_mean_percentile", "jpz_percentile_unit", "jpz_registered", "jpz_sat", "jpz_admitted", "jpz_metric_family",
        "mz_mean_score_pct", "mz_school_mean_percentile", "mz_school_mean_percentile_method", "mz_school_mean_percentile_reference",
        "slope_mz_mean_score_pct_per_year", "slope_mz_school_mean_percentile_per_year", "mz_school_mean_percentile_n_schools", "mz_candidates", "mz_pass_rate",
        "match_status", "source", "modern_actual_score_value", "modern_actual_score_unit", "modern_capacity",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = None
    return out[keep]


def _build_model(archive_dir: Path) -> dict[str, Any]:
    reports_dir = archive_dir / "reports"
    normalized_dir = archive_dir / "normalized"
    manifest = _read_json(archive_dir / "manifest.json", {}) or {}
    methodology = _read_json(reports_dir / "methodology.json", {}) or {}

    cohort = _read_csv(reports_dir / "cohort_matched" / "cohort_component_panel.csv")
    scenario_intake = _read_csv(reports_dir / "cohort_matched" / "scenario_intake_vs_mz_outcomes.csv")
    expected = _read_csv(reports_dir / "cohort_matched" / "expected_vs_observed_association.csv")
    expected_meta = _read_json(reports_dir / "cohort_matched" / "expected_vs_observed_metadata.json", {}) or {}
    jpz_trends = _read_csv(reports_dir / "cross_year_descriptive" / "jpz_school_component_trends.csv")
    mz_trends = _read_csv(reports_dir / "cross_year_descriptive" / "mz_school_component_trends.csv")
    jpz_components = _read_csv(normalized_dir / "jpz_components.csv")
    mz_components = _read_csv(normalized_dir / "mz_components.csv")
    school_dim = _read_csv(normalized_dir / "school_dimension.csv")

    if not expected.empty:
        if "component" in expected.columns:
            expected["component"] = expected["component"].map(_label_component)
        if "school_key" in expected.columns and "redizo" in expected.columns:
            expected["redizo"] = expected["redizo"].where(expected["redizo"].notna(), expected["school_key"].map(_redizo_from_school_key))

    if not scenario_intake.empty:
        if "outcome_component" in scenario_intake.columns:
            scenario_intake["outcome_component"] = scenario_intake["outcome_component"].map(_label_component)
        if "school_key" in scenario_intake.columns and "redizo" in scenario_intake.columns:
            scenario_intake["redizo"] = scenario_intake["redizo"].where(scenario_intake["redizo"].notna(), scenario_intake["school_key"].map(_redizo_from_school_key))

    if not jpz_trends.empty and "component" in jpz_trends.columns:
        jpz_trends["component"] = jpz_trends["component"].map(_label_component)

    if not mz_trends.empty and "component" in mz_trends.columns:
        mz_trends["component"] = mz_trends["component"].map(_label_component)

    scenario_cols = [
        "school_key", "redizo", "school_name_raw", "school_name_raw_jpz",
        "address_raw", "address_raw_jpz", "school_type", "outcome_component",
        "entry_year", "graduation_year", "jpz_mean_percentile",
        "synthetic_admitted_intake_selectivity_percentile",
        "outcome_mz_school_mean_percentile", "outcome_mz_mean_score_pct",
        "capacity_throughput_proxy_candidates", "mz_participation_rate_vs_cj",
    ]

    expected_cols = [
        "school_key", "redizo", "component", "entry_year", "graduation_year",
        "mz_mean_score_pct_expected", "residual_pp",
    ]

    jpz_cols = [
        "school_key", "redizo", "school_name_raw", "address_raw",
        "school_type", "component", "mean_jpz_percentile", "n_years",
    ]

    mz_cols = [
        "school_key", "redizo", "school_name_raw", "address_raw",
        "school_type", "component", "mz_school_mean_percentile",
        "n_years", "max_mz_candidates", "mean_mz_participation_rate_vs_cj",
    ]

    sources = manifest.get("sources", []) if isinstance(manifest, dict) else []
    unavailable = [s for s in sources if str(s.get("status")) == "unavailable"]
    downloaded = [s for s in sources if str(s.get("status")) == "downloaded"]

    school_type_values: list[str] = []
    for frame in (jpz_components, mz_components, cohort, expected, jpz_trends, mz_trends, school_dim):
        if frame.empty or "school_type" not in frame.columns:
            continue
        school_type_values.extend([str(x) for x in frame["school_type"].dropna().astype(str).tolist() if str(x).strip()])
    school_type_counts = pd.Series(school_type_values).value_counts().to_dict() if school_type_values else {}

    def _coverage(df: pd.DataFrame, key: str) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "schools": 0, "years": []}
        years = sorted({int(x) for x in pd.to_numeric(df.get(key, pd.Series(dtype="float64")), errors="coerce").dropna().astype(int).tolist()})
        return {"rows": int(len(df)), "schools": int(df.get("school_key", pd.Series(dtype="string")).nunique()), "years": years}

    return _strip_urls(
        {
            "archive": {"path": str(archive_dir), "freeze_id": manifest.get("freeze_id"), "created_at": manifest.get("created_at")},
            "availability": {
                "sources": [
                    {
                        "source_id": s.get("source_id"),
                        "dataset": s.get("dataset"),
                        "year": s.get("year"),
                        "kind": s.get("kind"),
                        "status": s.get("status"),
                        "reason": s.get("reason"),
                    }
                    for s in sources
                ],
                "explicit_unavailable": [{"source_id": s.get("source_id"), "reason": s.get("reason") or s.get("unavailable_reason")} for s in unavailable],
            },
            "coverage": {
                "cohort_rows": int(len(cohort)),
                "downloaded_sources": int(len(downloaded)),
                "unavailable_sources": int(len(unavailable)),
                "expected_rows": int(len(expected)),
                "scenario_intake_rows": int(len(scenario_intake)),
                "jpz": _coverage(jpz_components, "entry_year"),
                "mz": _coverage(mz_components, "year"),
                "school_type_counts": {str(k): int(v) for k, v in school_type_counts.items()},
            },
            "scenario_intake": {"rows": _to_columnar(scenario_intake, scenario_cols)},
            "cross_year": {"jpz_rows": _to_columnar(jpz_trends, jpz_cols), "mz_rows": _to_columnar(mz_trends, mz_cols)},
            "expected_observed": {"rows": _to_columnar(expected, expected_cols), "metadata": expected_meta},
            "methodology": {
                "language": methodology.get("language"),
                "metric_units": methodology.get("metric_units"),
                "notes": methodology.get("notes", []),
            },
        }
    )


def write_archive_dashboard(archive_dir: str | Path, output_path: str | Path | None = None) -> Path:
    arch = Path(archive_dir).resolve()
    out = (arch / "reports" / "archive_dashboard.html") if output_path is None else Path(output_path)
    if not out.is_absolute():
        out = arch / out
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = _build_model(arch)
    data_json = _safe_json(model)
    definitions_json = _safe_json([
        "Selectivity percentile (synthetic) is a non-causal scenario proxy in 0–100 percentile space (not latent z).",
        "MZ throughput proxy (max candidates) is bubble size where available; it is not seat capacity.",
        "Multi-year tab uses cross-year JPZ/MZ descriptive rows across all available years, independent of cohort matching.",
        "Expected/residual are joined by school_key + component + entry_year + graduation_year; AJ expected/residual are blank by design.",
        "MZ participation rate (relative to CJ, %) = component MZ candidates / CJ MZ candidates for the same school_key + school_type + graduation_year (variant preference jap>j, candidates summed across duplicate rows sharing the key). It is a proxy using CJ candidates as a stand-in for the compulsory-cohort size, not a true enrollment/graduation registry figure, and it may include repeat, private, or adult candidates who sat the exam without belonging to the standard cohort. CJ rows are expected to be ~100% (a data-quality self-check); M and AJ rows are typically lower since those components are optional. The rate is null when CJ candidates are missing or zero.",
    ])

    html_text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Archive dashboard</title>
<style>body{{font:14px system-ui;margin:0;background:#f6f8fb}}main,header{{max-width:1500px;margin:auto;padding:16px}}.tabs{{display:flex;gap:8px;flex-wrap:wrap}}.tab-btn{{padding:7px 12px;border:1px solid #dbe3ec;border-radius:999px;background:#fff}}.tab-btn[aria-selected="true"]{{background:#2158d6;color:#fff}}.tab-panel{{display:none}}.tab-panel.active{{display:block}}.panel{{background:#fff;border:1px solid #dbe3ec;border-radius:12px;padding:12px;margin-top:10px}}.toolbar{{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}label{{display:grid;gap:4px}}input,select{{padding:8px;border:1px solid #dbe3ec;border-radius:10px}}svg{{max-width:100%;height:auto;display:block}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #dbe3ec;text-align:left}}.pager{{display:flex;align-items:center;gap:10px;margin:8px 0}}.pager button{{padding:6px 10px;border:1px solid #dbe3ec;border-radius:8px;background:#fff}}</style></head><body>
<header><h1>Local archive dashboard</h1><div id="archive-meta"></div></header><main>
<nav class="tabs"><button class="tab-btn" data-tab="summary" aria-selected="true">Summary</button><button class="tab-btn" data-tab="cohort" aria-selected="false">Cohort view</button><button class="tab-btn" data-tab="multiyear" aria-selected="false">Multi-year view</button><button class="tab-btn" data-tab="methods" aria-selected="false">Methods</button><button class="tab-btn" data-tab="sources" aria-selected="false">Sources</button></nav>
<section class="tab-panel active" id="tab-summary"><div class="panel"><h2>Summary</h2><div id="summary-kpis"></div></div></section>
<section class="tab-panel" id="tab-cohort"><div class="panel"><h2>Cohort view</h2><div class="toolbar"><label>Search <input id="cohort-search" type="search" placeholder="School, REDIZO, address"></label><label>School type <select id="cohort-school-type"></select></label><label>Component <select id="cohort-component"></select></label><label>Graduation year <select id="cohort-grad-year"></select></label><label>Rows per page <select id="cohort-page-size"><option value="10">10</option><option value="25" selected>25</option><option value="50">50</option><option value="100">100</option><option value="all">All</option></select></label></div><div class="pager"><button id="cohort-page-prev" type="button">Prev</button><span id="cohort-page-info"></span><button id="cohort-page-next" type="button">Next</button></div><div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px"><div class="panel"><h3>JPZ percentile → MZ percentile</h3><svg id="cohort-scatter-jpz" viewBox="0 0 760 440"></svg></div><div class="panel"><h3>Selectivity percentile (synthetic) → MZ percentile</h3><svg id="cohort-scatter-selectivity" viewBox="0 0 760 440"></svg></div></div><table><thead><tr id="cohort-head"></tr></thead><tbody id="cohort-body"></tbody></table></div></section>
<section class="tab-panel" id="tab-multiyear"><div class="panel"><h2>Multi-year view</h2><div class="toolbar"><label>Search <input id="multi-search" type="search" placeholder="School, REDIZO, address"></label><label>School type <select id="multi-school-type"></select></label><label>Component <select id="multi-component"></select></label><label>Rows per page <select id="multi-page-size"><option value="10">10</option><option value="25" selected>25</option><option value="50">50</option><option value="100">100</option><option value="all">All</option></select></label></div><div class="pager"><button id="multi-page-prev" type="button">Prev</button><span id="multi-page-info"></span><button id="multi-page-next" type="button">Next</button></div><div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px"><div class="panel"><h3>Average JPZ percentile → average MZ percentile</h3><svg id="multi-scatter-jpz" viewBox="0 0 760 440"></svg></div><div class="panel"><h3>Average selectivity percentile (synthetic) → average MZ percentile</h3><svg id="multi-scatter-selectivity" viewBox="0 0 760 440"></svg></div></div><table><thead><tr id="multi-head"></tr></thead><tbody id="multi-body"></tbody></table></div></section>
<section class="tab-panel" id="tab-methods"><div class="panel"><h2>Methods</h2><ul id="definitions"></ul></div></section>
<section class="tab-panel" id="tab-sources"><div class="panel"><h2>Sources</h2><div id="source-list"></div></div></section>
</main><script id="dashboard-data" type="application/json">{data_json}</script><script id="dashboard-js" type="text/javascript">
const DATA=JSON.parse(document.getElementById('dashboard-data').textContent);
function unpack(t){{if(!t)return[];if(Array.isArray(t))return t;const c=t.cols||[],r=t.rows||[];return r.map(a=>{{if(!Array.isArray(a))return a;const o={{}};for(let i=0;i<c.length;i++)o[c[i]]=a[i];return o;}});}}
if(DATA.expected_observed?.rows)DATA.expected_observed.rows=unpack(DATA.expected_observed.rows);
if(DATA.scenario_intake?.rows)DATA.scenario_intake.rows=unpack(DATA.scenario_intake.rows);
if(DATA.cross_year?.jpz_rows)DATA.cross_year.jpz_rows=unpack(DATA.cross_year.jpz_rows);
if(DATA.cross_year?.mz_rows)DATA.cross_year.mz_rows=unpack(DATA.cross_year.mz_rows);
const defs={definitions_json};const el=id=>document.getElementById(id);const esc=v=>String(v??'—').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const asNum=v=>{{if(v===null||v===undefined)return null;if(typeof v==='string'){{const s=v.trim();if(!s||/^(na|n\/a|null|undefined|nan|\.)$/i.test(s))return null;}}const n=Number(v);return Number.isFinite(n)?n:null;}};const fmt=(v,d=1)=>{{const n=asNum(v);return n===null?'—':n.toFixed(d);}};const intFmt=v=>{{const n=asNum(v);return n===null?'—':new Intl.NumberFormat('en-US').format(n).replace(/,/g,' ');}};const comp=v=>({{CJ:'CJ',M:'M',AJ:'AJ'}}[String(v||'')]||String(v||''));
function expectedMap(){{const m=new Map();(DATA.expected_observed?.rows||[]).forEach(r=>m.set(schoolKeyOf(r)+'|'+(r.component||'')+'|'+(r.entry_year||'')+'|'+(r.graduation_year||''),r));return m;}}
function schoolKeyOf(r){{return r&&r.school_key?String(r.school_key):('redizo:'+(r&&r.redizo||''));}}
function cohortRows(){{const em=expectedMap();return (DATA.scenario_intake?.rows||[]).map(r=>{{const c=String(r.outcome_component||'');const e=em.get(schoolKeyOf(r)+'|'+c+'|'+(r.entry_year||'')+'|'+(r.graduation_year||''));return {{redizo:r.redizo,school_name:r.school_name_raw_jpz||r.school_name_raw||'',address:r.address_raw_jpz||r.address_raw||'',school_type:r.school_type,component:c,entry_year:r.entry_year,graduation_year:r.graduation_year,jpz:asNum(r.jpz_mean_percentile),sel:asNum(r.synthetic_admitted_intake_selectivity_percentile),mz:asNum(r.outcome_mz_school_mean_percentile),mzScore:asNum(r.outcome_mz_mean_score_pct),exp:(c==='AJ'?null:(e?asNum(e.mz_mean_score_pct_expected):null)),res:(c==='AJ'?null:(e?asNum(e.residual_pp):null)),proxy:asNum(r.capacity_throughput_proxy_candidates),part:asNum(r.mz_participation_rate_vs_cj)!==null?asNum(r.mz_participation_rate_vs_cj)*100:null}};}});}}
function selectivityAgg(){{const rows=DATA.scenario_intake?.rows||[];const acc=new Map();rows.forEach(r=>{{const v=asNum(r.synthetic_admitted_intake_selectivity_percentile);if(v===null)return;const key=[schoolKeyOf(r),r.school_type||'',r.outcome_component||''].join('|');const cur=acc.get(key)||{{sum:0,n:0}};cur.sum+=v;cur.n+=1;acc.set(key,cur);}});const out=new Map();acc.forEach((v,k)=>out.set(k,v.sum/v.n));return out;}}
function multiRows(){{const j=(DATA.cross_year?.jpz_rows||[]).map(r=>({{...r,component:comp(r.component),jpz:asNum(r.mean_jpz_percentile),n_j:asNum(r.n_years)}}));const m=(DATA.cross_year?.mz_rows||[]).map(r=>({{...r,component:comp(r.component),mz:asNum(r.mz_school_mean_percentile),n_m:asNum(r.n_years),px:asNum(r.max_mz_candidates)}}));const jm=new Map(j.map(r=>[[r.school_key||'',r.school_type||'',r.component||''].join('|'),r]));const selMap=selectivityAgg();return m.map(r=>{{const jr=jm.get([r.school_key||'',r.school_type||'',r.component||''].join('|'));const cov=Math.max(asNum(jr?.n_j)||0,asNum(r.n_m)||0)||null;const selKey=[schoolKeyOf(r),r.school_type||'',r.component||''].join('|');return {{redizo:r.redizo||jr?.redizo,school_name:r.school_name_raw||jr?.school_name_raw||'',address:r.address_raw||jr?.address_raw||'',school_type:r.school_type,component:r.component,jpz:jr?.jpz??null,mz:r.mz,sel:selMap.has(selKey)?selMap.get(selKey):null,n_j:jr?.n_j??null,n_m:r.n_m,proxy:r.px,cov,size:(r.px!==null&&r.px>0)?r.px:cov,part:asNum(r.mean_mz_participation_rate_vs_cj)!==null?asNum(r.mean_mz_participation_rate_vs_cj)*100:null}};}});}}
function ticks(mi,ma,n=6){{if(mi===ma)return[mi];const a=[];const s=(ma-mi)/(n-1);for(let i=0;i<n;i++)a.push(mi+i*s);return a;}}
const SCHOOL_TYPE_PALETTE=['#2158d6','#d62158','#21a366','#d68821','#8821d6','#21b6d6','#c2410c','#4d7c0f','#0f766e','#7c3aed'];
function schoolTypeColor(t){{const key=String(t||'UNKNOWN');let h=0;for(let i=0;i<key.length;i++)h=(h*31+key.charCodeAt(i))>>>0;return SCHOOL_TYPE_PALETTE[h%SCHOOL_TYPE_PALETTE.length];}}
function drawScatter(id,rows,xk,yk,sk,xlab,ylab,slab,diag){{const svg=el(id),W=760,H=440,m={{l:70,r:20,t:20,b:60}};if(!svg)return;const pts=rows.filter(r=>asNum(r[xk])!==null&&asNum(r[yk])!==null&&asNum(r[sk])!==null&&asNum(r[sk])>0);if(!pts.length){{svg.innerHTML='';return;}}const xs=pts.map(r=>asNum(r[xk])),ys=pts.map(r=>asNum(r[yk]));const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const sx=v=>m.l+((v-xmin)/(xmax-xmin||1))*(W-m.l-m.r),sy=v=>H-m.b-((v-ymin)/(ymax-ymin||1))*(H-m.t-m.b);const sv=pts.map(r=>asNum(r[sk])).filter(v=>v!==null&&v>0),smin=sv.length?Math.min(...sv):null,smax=sv.length?Math.max(...sv):null;const rr=v=>{{const n=asNum(v);if(n===null||n<=0||smin===null||smax===null)return 4;return 3+((n-smin)/(smax-smin||1))*10;}};let h=`<rect x='0' y='0' width='${{W}}' height='${{H}}' fill='#fff'/><line x1='${{m.l}}' y1='${{H-m.b}}' x2='${{W-m.r}}' y2='${{H-m.b}}' stroke='#667'/><line x1='${{m.l}}' y1='${{m.t}}' x2='${{m.l}}' y2='${{H-m.b}}' stroke='#667'/>`;ticks(xmin,xmax).forEach(t=>{{const x=sx(t);h+=`<line x1='${{x}}' y1='${{H-m.b}}' x2='${{x}}' y2='${{H-m.b+5}}' stroke='#667'/><text x='${{x}}' y='${{H-m.b+20}}' text-anchor='middle' font-size='11'>${{t.toFixed(1)}}</text>`;}});ticks(ymin,ymax).forEach(t=>{{const y=sy(t);h+=`<line x1='${{m.l-5}}' y1='${{y}}' x2='${{m.l}}' y2='${{y}}' stroke='#667'/><text x='${{m.l-8}}' y='${{y+4}}' text-anchor='end' font-size='11'>${{t.toFixed(1)}}</text>`;}});h+=`<text x='${{(m.l+W-m.r)/2}}' y='${{H-10}}' text-anchor='middle' font-size='12'>${{esc(xlab)}}</text><text transform='translate(16 ${{(m.t+H-m.b)/2}}) rotate(-90)' text-anchor='middle' font-size='12'>${{esc(ylab)}}</text><text x='${{W-m.r}}' y='${{m.t+12}}' text-anchor='end' font-size='11'>${{esc(slab)}}</text>`;if(diag){{h+=`<clipPath id='${{id}}-plotclip'><rect x='${{m.l}}' y='${{m.t}}' width='${{W-m.l-m.r}}' height='${{H-m.t-m.b}}'/></clipPath><line x1='${{sx(0)}}' y1='${{sy(0)}}' x2='${{sx(100)}}' y2='${{sy(100)}}' stroke='#94a3b8' stroke-width='1.5' stroke-dasharray='5,4' opacity='0.7' clip-path='url(#${{id}}-plotclip)'/>`;}}pts.forEach(r=>{{const col=schoolTypeColor(r.school_type);h+=`<circle cx='${{sx(asNum(r[xk]))}}' cy='${{sy(asNum(r[yk]))}}' r='${{rr(r[sk])}}' fill='${{col}}' fill-opacity='0.55' stroke='${{col}}'/>`;}});svg.innerHTML=h;}}
function addOptions(id,vals){{const s=el(id);s.innerHTML='';['All',...vals].forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o);}});}}
function blob(r){{return[r.redizo,r.school_name,r.address,r.school_type,r.component,r.graduation_year].map(x=>String(x||'')).join(' ').toLowerCase();}}
const pageState={{cohort:1,multi:1}};
const sortState={{cohort:{{key:null,dir:1}},multi:{{key:null,dir:1}}}};
const COHORT_COLS=[['Graduation year','graduation_year'],['Entry year','entry_year'],['REDIZO','redizo'],['School','school_name'],['Address','address'],['Type','school_type'],['Component','component'],['JPZ percentile','jpz'],['Selectivity percentile (synthetic)','sel'],['MZ percentile','mz'],['MZ score (%)','mzScore'],['Expected MZ percentile','exp'],['Residual (pp)','res'],['MZ throughput proxy (max candidates)','proxy'],['MZ participation rate (relative to CJ, %)','part']];
const MULTI_COLS=[['REDIZO','redizo'],['School','school_name'],['Address','address'],['Type','school_type'],['Component','component'],['Avg JPZ percentile','jpz'],['Avg selectivity percentile (synthetic)','sel'],['Avg MZ percentile','mz'],['JPZ years','n_j'],['MZ years','n_m'],['MZ throughput proxy (max candidates)','proxy'],['Coverage years (max JPZ/MZ)','cov'],['Avg MZ participation rate (relative to CJ, %)','part']];
function sortRows(prefix,rows){{const st=sortState[prefix];if(!st.key)return rows;const key=st.key,dir=st.dir;return rows.slice().sort((a,b)=>{{const av=a[key],bv=b[key];const an=asNum(av),bn=asNum(bv);if(an!==null&&bn!==null)return(an-bn)*dir;return String(av??'').localeCompare(String(bv??''))*dir;}});}}
function headerHtml(prefix,cols){{const st=sortState[prefix];return cols.map(([label,key])=>{{const arrow=st.key===key?(st.dir===1?' \u25b2':' \u25bc'):'';return `<th data-key="${{key}}" data-prefix="${{prefix}}" style="cursor:pointer;user-select:none">${{esc(label)}}${{arrow}}</th>`;}}).join('');}}
function attachSortHandlers(prefix,headEl,renderFn){{headEl.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{{const key=th.dataset.key;const st=sortState[prefix];if(st.key===key){{st.dir=-st.dir;}}else{{st.key=key;st.dir=1;}}pageState[prefix]=1;renderFn();}}));}}
function pageSizeOf(prefix){{const v=el(prefix+'-page-size').value;return v==='all'?Infinity:parseInt(v,10);}}
function paginate(prefix,rows){{const size=pageSizeOf(prefix);const total=rows.length;const pages=Math.max(1,Math.ceil(total/size));if(pageState[prefix]>pages)pageState[prefix]=pages;if(pageState[prefix]<1)pageState[prefix]=1;const page=pageState[prefix];const startIdx=size===Infinity?0:(page-1)*size;const endIdx=size===Infinity?total:Math.min(total,startIdx+size);const info=el(prefix+'-page-info');if(info)info.textContent=total===0?'Showing 0-0 of 0':`Showing ${{startIdx+1}}-${{endIdx}} of ${{total}}`;const prevBtn=el(prefix+'-page-prev'),nextBtn=el(prefix+'-page-next');if(prevBtn)prevBtn.disabled=page<=1;if(nextBtn)nextBtn.disabled=page>=pages;return rows.slice(startIdx,endIdx);}}
function renderCohort(){{let rows=cohortRows();const q=(el('cohort-search').value||'').toLowerCase().trim(),st=el('cohort-school-type').value,c=el('cohort-component').value,g=el('cohort-grad-year').value;if(q)rows=rows.filter(r=>blob(r).includes(q));if(st!=='All')rows=rows.filter(r=>String(r.school_type)===st);if(c!=='All')rows=rows.filter(r=>String(r.component)===c);if(g!=='All')rows=rows.filter(r=>String(r.graduation_year)===g);drawScatter('cohort-scatter-jpz',rows,'jpz','mz','proxy','JPZ percentile (0-100)','MZ school mean percentile (0-100)','Bubble size = MZ throughput proxy (max candidates)',true);drawScatter('cohort-scatter-selectivity',rows,'sel','mz','proxy','Selectivity percentile (synthetic, 0-100)','MZ school mean percentile (0-100)','Bubble size = MZ throughput proxy (max candidates)');rows=sortRows('cohort',rows);el('cohort-head').innerHTML=headerHtml('cohort',COHORT_COLS);attachSortHandlers('cohort',el('cohort-head'),renderCohort);const pageRows=paginate('cohort',rows);el('cohort-body').innerHTML=pageRows.map(r=>`<tr><td>${{esc(r.graduation_year)}}</td><td>${{esc(r.entry_year)}}</td><td>${{esc(r.redizo)}}</td><td>${{esc(r.school_name)}}</td><td>${{esc(r.address)}}</td><td>${{esc(r.school_type)}}</td><td>${{esc(r.component)}}</td><td>${{fmt(r.jpz,1)}}</td><td>${{fmt(r.sel,1)}}</td><td>${{fmt(r.mz,1)}}</td><td>${{fmt(r.mzScore,1)}}</td><td>${{fmt(r.exp,1)}}</td><td>${{fmt(r.res,2)}}</td><td>${{intFmt(r.proxy)}}</td><td>${{fmt(r.part,1)}}</td></tr>`).join('');}}
function renderMulti(){{let rows=multiRows();const q=(el('multi-search').value||'').toLowerCase().trim(),st=el('multi-school-type').value,c=el('multi-component').value;if(q)rows=rows.filter(r=>blob(r).includes(q));if(st!=='All')rows=rows.filter(r=>String(r.school_type)===st);if(c!=='All')rows=rows.filter(r=>String(r.component)===c);const hasProxy=rows.some(r=>asNum(r.proxy)!==null&&asNum(r.proxy)>0);drawScatter('multi-scatter-jpz',rows,'jpz','mz','size','Average JPZ percentile across all available JPZ years','Average MZ school mean percentile across all available MZ years',hasProxy?'Bubble size = MZ throughput proxy (max candidates)':'Bubble size = cross-year coverage count (max JPZ/MZ years)',true);drawScatter('multi-scatter-selectivity',rows,'sel','mz','size','Average selectivity percentile (synthetic, 0-100)','Average MZ school mean percentile across all available MZ years',hasProxy?'Bubble size = MZ throughput proxy (max candidates)':'Bubble size = cross-year coverage count (max JPZ/MZ years)');rows=sortRows('multi',rows);el('multi-head').innerHTML=headerHtml('multi',MULTI_COLS);attachSortHandlers('multi',el('multi-head'),renderMulti);const pageRows=paginate('multi',rows);el('multi-body').innerHTML=pageRows.map(r=>`<tr><td>${{esc(r.redizo)}}</td><td>${{esc(r.school_name)}}</td><td>${{esc(r.address)}}</td><td>${{esc(r.school_type)}}</td><td>${{esc(r.component)}}</td><td>${{fmt(r.jpz,1)}}</td><td>${{fmt(r.sel,1)}}</td><td>${{fmt(r.mz,1)}}</td><td>${{intFmt(r.n_j)}}</td><td>${{intFmt(r.n_m)}}</td><td>${{intFmt(r.proxy)}}</td><td>${{intFmt(r.cov)}}</td><td>${{fmt(r.part,1)}}</td></tr>`).join('');}}
function boot(){{el('archive-meta').textContent=`${{DATA.archive?.freeze_id||''}} · ${{DATA.archive?.created_at||''}}`;el('summary-kpis').innerHTML=[['Cohort rows',DATA.coverage?.cohort_rows],['Expected rows',DATA.coverage?.expected_rows],['Scenario rows',DATA.coverage?.scenario_intake_rows],['JPZ rows',DATA.coverage?.jpz?.rows],['MZ rows',DATA.coverage?.mz?.rows]].map(([k,v])=>`<div><strong>${{esc(k)}}:</strong> ${{intFmt(v)}}</div>`).join('');el('definitions').innerHTML=defs.map(d=>`<li>${{esc(d)}}</li>`).join('');el('source-list').innerHTML=(DATA.availability?.sources||[]).map(s=>`<div><strong>${{esc(s.source_id)}}</strong> — ${{esc(s.status)}} ${{s.reason?`(${{esc(s.reason)}})`:''}}</div>`).join('');const c=cohortRows();addOptions('cohort-school-type',[...new Set(c.map(r=>String(r.school_type||'')).filter(Boolean))].sort());addOptions('cohort-component',['CJ','M','AJ']);addOptions('cohort-grad-year',[...new Set(c.map(r=>String(r.graduation_year||'')).filter(Boolean))].sort());const m=multiRows();addOptions('multi-school-type',[...new Set(m.map(r=>String(r.school_type||'')).filter(Boolean))].sort());addOptions('multi-component',['CJ','M','AJ']);['cohort-search','cohort-school-type','cohort-component','cohort-grad-year'].forEach(id=>el(id).addEventListener('input',()=>{{pageState.cohort=1;renderCohort();}}));['cohort-school-type','cohort-component','cohort-grad-year'].forEach(id=>el(id).addEventListener('change',()=>{{pageState.cohort=1;renderCohort();}}));el('cohort-page-size').addEventListener('change',()=>{{pageState.cohort=1;renderCohort();}});el('cohort-page-prev').addEventListener('click',()=>{{pageState.cohort=Math.max(1,pageState.cohort-1);renderCohort();}});el('cohort-page-next').addEventListener('click',()=>{{pageState.cohort=pageState.cohort+1;renderCohort();}});['multi-search','multi-school-type','multi-component'].forEach(id=>el(id).addEventListener('input',()=>{{pageState.multi=1;renderMulti();}}));['multi-school-type','multi-component'].forEach(id=>el(id).addEventListener('change',()=>{{pageState.multi=1;renderMulti();}}));el('multi-page-size').addEventListener('change',()=>{{pageState.multi=1;renderMulti();}});el('multi-page-prev').addEventListener('click',()=>{{pageState.multi=Math.max(1,pageState.multi-1);renderMulti();}});el('multi-page-next').addEventListener('click',()=>{{pageState.multi=pageState.multi+1;renderMulti();}});document.querySelectorAll('.tab-btn').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('.tab-btn').forEach(x=>x.setAttribute('aria-selected','false'));document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));b.setAttribute('aria-selected','true');el('tab-'+b.dataset.tab).classList.add('active');}}));renderCohort();renderMulti();}}
boot();</script></body></html>'''
    out.write_text(html_text, encoding="utf-8")
    return out
