from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from gymnazium_value_added.model import AnalysisConfig

LOGGER = logging.getLogger(__name__)


def _ridge_fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(np.clip(w, 1e-9, None))[:, None]
    xw = x_aug * sw
    yw = y * sw[:, 0]
    reg = np.eye(xw.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    return x_aug @ beta


def _prepare_school_panel(
    admissions: pd.DataFrame,
    maturita: pd.DataFrame,
    cohort_lag_years: int,
) -> pd.DataFrame:
    adm = admissions.copy()
    mat = maturita.copy()

    adm = adm.dropna(subset=["school_id", "year"])
    if "applications" not in adm.columns:
        adm["applications"] = pd.NA
    if "capacity" not in adm.columns:
        adm["capacity"] = pd.NA
    adm["applications"] = pd.to_numeric(adm["applications"], errors="coerce")
    adm["capacity"] = pd.to_numeric(adm["capacity"], errors="coerce")
    sel_present = adm[["applications", "capacity"]].notna().any(axis=1)
    inconsistent = sel_present & ~(adm["applications"].notna() & adm["capacity"].notna())
    adm = adm[~inconsistent].copy()
    adm = adm[adm["capacity"].isna() | (adm["capacity"] > 0)]
    adm["entry_year"] = adm["year"].astype(int)
    if "selection_metric" in adm.columns:
        adm["selection_metric"] = pd.to_numeric(adm["selection_metric"], errors="coerce")
    elif {"applications", "capacity"}.issubset(adm.columns):
        adm["selection_metric"] = adm["applications"] / adm["capacity"]
    else:
        adm["selection_metric"] = pd.NA
    adm["selection_metric_observed"] = adm.get("selection_metric_observed", adm["selection_metric"].notna())
    adm["selectivity_ratio"] = adm["selection_metric"]

    adm_agg = (
        adm.groupby(["school_id", "school_name", "entry_year"], dropna=False)
        .agg(
            applications=("applications", lambda s: s.sum(min_count=1)),
            capacity=("capacity", lambda s: s.sum(min_count=1)),
            selection_metric=("selection_metric", "mean"),
            selection_metric_observed=("selection_metric_observed", "max"),
            avg_admission_score=("avg_admission_score", "mean"),
        )
        .reset_index()
    )

    mat = mat.dropna(subset=["school_id", "year", "candidates"])
    mat["grad_year"] = mat["year"].astype(int)
    mat["entry_year"] = mat["grad_year"] - cohort_lag_years

    use_mean_score = mat["mean_score"].notna().any()
    if not use_mean_score:
        LOGGER.warning("mean_score is unavailable; using pass_rate as the outcome metric.")
    mat["outcome_value"] = mat["mean_score"] if use_mean_score else mat["pass_rate"]

    mat_agg = (
        mat.groupby(["school_id", "school_name", "entry_year", "grad_year", "subject"], dropna=False)
        .agg(
            candidates=("candidates", "sum"),
            outcome=("outcome_value", "mean"),
        )
        .reset_index()
    )
    mat_agg["outcome_metric"] = "mean_score" if use_mean_score else "pass_rate"

    panel = mat_agg.merge(
        adm_agg,
        on=["school_id", "school_name", "entry_year"],
        how="left",
        suffixes=("_mat", "_adm"),
    )
    panel = panel.dropna(subset=["outcome", "candidates"])
    return panel


def _design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame, list[str]]:
    base = panel.copy()
    fallbacks: list[str] = []

    use_selectivity = base["selection_metric"].notna().any()
    use_score = base["avg_admission_score"].notna().any()

    feature_blocks = []
    feature_names: list[str] = []

    if use_selectivity:
        median_sel = float(base["selection_metric"].median())
        base["selection_metric"] = base["selection_metric"].fillna(median_sel)
        feature_blocks.append(base[["selection_metric"]].reset_index(drop=True))
        feature_names.append("selection_metric")
    else:
        fallbacks.append("missing_selectivity")

    if use_score:
        median_score = float(base["avg_admission_score"].median())
        base["avg_admission_score"] = base["avg_admission_score"].fillna(median_score)
        feature_blocks.append(base[["avg_admission_score"]].reset_index(drop=True))
        feature_names.append("avg_admission_score")
    else:
        fallbacks.append("missing_avg_admission_score")

    d_subject = pd.get_dummies(base["subject"].astype(str), prefix="subject", drop_first=True)
    d_year = pd.get_dummies(base["grad_year"].astype(str), prefix="year", drop_first=True)
    feature_blocks.extend([d_subject.reset_index(drop=True), d_year.reset_index(drop=True)])
    feature_names.extend(list(d_subject.columns))
    feature_names.extend(list(d_year.columns))

    X_df = pd.concat(feature_blocks, axis=1)
    return X_df.to_numpy(dtype=float), feature_names, base, fallbacks


def analyze_value_added(
    admissions: pd.DataFrame,
    maturita: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = _prepare_school_panel(admissions, maturita, config.cohort_lag_years)
    if panel.empty:
        raise ValueError("No data available for analysis after joining admissions and maturita tables.")

    X, feature_names, base, fallbacks = _design_matrix(panel)
    y = base["outcome"].to_numpy(dtype=float)
    w = base["candidates"].to_numpy(dtype=float)

    pred = _ridge_fit_predict(X, y, w, config.ridge_alpha)
    base = base.copy()
    base["predicted_outcome"] = pred
    base["residual"] = base["outcome"] - base["predicted_outcome"]

    school = (
        base.groupby(["school_id", "school_name"], dropna=False)
        .agg(
            cohorts=("entry_year", "nunique"),
            total_candidates=("candidates", "sum"),
            mean_selection_metric=("selection_metric", "mean"),
            selection_metric_observed=("selection_metric_observed", "mean"),
            selection_metric_missing=("selection_metric_observed", lambda s: float((~pd.Series(s).fillna(False)).any())),
            mean_admission_score=("avg_admission_score", "mean"),
            observed_outcome=("outcome", "mean"),
            expected_outcome=("predicted_outcome", "mean"),
            value_added=("residual", "mean"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(config.random_seed)
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    for _, row in school.iterrows():
        sdf = base[base["school_id"] == row["school_id"]]
        vals = sdf["residual"].to_numpy()
        if len(vals) < 2:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            continue
        boots = []
        for _ in range(config.bootstrap_iterations):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(float(np.mean(sample)))
        ci_lo.append(float(np.quantile(boots, 0.025)))
        ci_hi.append(float(np.quantile(boots, 0.975)))

    school["value_added_ci_low"] = ci_lo
    school["value_added_ci_high"] = ci_hi
    school["sufficient_data"] = school["total_candidates"] >= config.min_cohort_size
    school["quality_flag"] = np.where(
        school["sufficient_data"],
        "ok",
        f"low_candidates(<{config.min_cohort_size})",
    )
    if base["entry_year"].nunique() == 1 and int(school["school_id"].nunique()) < config.min_school_count:
        raise ValueError(
            f"Single-cohort report requires at least {config.min_school_count} schools; got {school['school_id'].nunique()}."
        )
    school = school.sort_values("value_added", ascending=False).reset_index(drop=True)

    meta: dict[str, Any] = {
        "analysis_type": "repeated_cross_section_school_level",
        "causal_warning": "This is not an individual-level causal estimate; it is a school-level adjusted association model.",
        "config": asdict(config),
        "features": feature_names,
        "fallbacks": fallbacks,
        "n_rows_panel": int(len(base)),
        "n_schools": int(school["school_id"].nunique()),
        "controls": ["subject_fixed_effect", "graduation_year_fixed_effect"],
        "outcome_definition": str(base["outcome_metric"].iloc[0]) if "outcome_metric" in base.columns and not base.empty else "unknown",
        "selection_metric_definition": "applications_per_capacity" if base["selection_metric"].notna().any() else "absent",
        "selection_metric_missing": bool(not base["selection_metric"].notna().any()),
        "avg_entry_score_available": bool(base["avg_admission_score"].notna().any()),
        "methodology": "Selection is modeled from observed applications/capacity when available; otherwise historic JPZ score-only cohorts keep selection_metric missing and are not median-imputed as observed.",
    }
    return school, meta
