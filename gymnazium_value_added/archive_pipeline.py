from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from gymnazium_value_added.archive_report import _build_unified_school_history, write_archive_dashboard
from gymnazium_value_added.config import dump_json, ensure_dir, load_json
from gymnazium_value_added.io import download_url, sha256_of_file


JPZ_HISTORIC_URL = "https://data.cermat.cz/files/files/JPZ/agregovana_data_skoly/JPZ{year}_skoly-skolobory_vysledky.xlsx"
JPZ_MODERN_URL = "https://data.cermat.cz/files/files/JPZ/agregovana_data_skoly/PZ{year}_kolo1_skolobory_{kind}.xlsx"
MZ_URL = "https://data.cermat.cz/files/files/MZ/agregovana_data_skoly/MZ{year}{variant}_SC_skolobory.xlsx"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _freeze_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _norm_text(value: object) -> str:
    if value is None:
        txt = ""
    else:
        try:
            if pd.isna(value):
                txt = ""
            else:
                txt = str(value)
        except Exception:
            txt = str(value)
    txt = txt.strip().lower()
    txt = "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))
    txt = re.sub(r"\s+", " ", txt)
    return txt


def _digits(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = re.search(r"\b\d{6,10}\b", str(value))
    return m.group(0) if m else None


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>"}:
        return None
    return text


def _norm_key_part(value: object) -> str:
    txt = _norm_text(value)
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")
    return txt


def _norm_tier(value: object) -> str:
    txt = _norm_text(value)
    return re.sub(r"[^a-z0-9]+", "", txt)


def _city_postcode(address: object) -> tuple[str | None, str | None]:
    text = _safe_text(address)
    if text is None:
        return None, None
    pm = re.search(r"\b(\d{3})\s?(\d{2})\b", text)
    postcode = f"{pm.group(1)} {pm.group(2)}" if pm else None
    city = None
    if pm and "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for part in reversed(parts):
            clean = re.sub(r"(?i)\bpsč\b", "", part)
            clean = re.sub(r"\b\d{3}\s?\d{2}\b", "", clean).strip(" -")
            if not clean:
                continue
            if re.search(r"\d", clean):
                continue
            if len(clean.split()) > 4:
                continue
            city = clean
            break
    return city, postcode


_TAXONOMY_ORDER = [
    "GY4",
    "GY6",
    "GY8",
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
    "TOTAL_AGGREGATE",
    "UNKNOWN",
]


_PROGRAMME_GROUP16_TAXONOMY_MAP: dict[str, str] = {
    "gy4": "GY4",
    "gy6": "GY6",
    "gy8": "GY8",
    "smo8": "GY4",
    "smo12": "GY6",
    "smo16": "GY8",
    "lyc": "LYC",
    "st1": "SOS_TECHNICAL",
    "st2": "SOS_TECHNICAL",
    "sek": "SOS_ECONOMIC",
    "shp": "SOS_HOTEL_BUSINESS",
    "shu": "SOS_PEDAGOGICAL_HUMANITIES",
    "sze": "SOS_AGRICULTURAL",
    "szd": "SOS_HEALTHCARE",
    "sum": "SOS_ARTS",
    "ute": "SOU_TECHNICAL",
    "uos": "SOU_OTHER",
    "nte": "NASTAVBA_TECHNICAL",
    "nos": "NASTAVBA_OTHER",
    "celkem": "TOTAL_AGGREGATE",
}


_SOS_MATCH_TYPES = {
    "SOS_TECHNICAL",
    "SOS_ECONOMIC",
    "SOS_HOTEL_BUSINESS",
    "SOS_PEDAGOGICAL_HUMANITIES",
    "SOS_AGRICULTURAL",
    "SOS_HEALTHCARE",
    "SOS_ARTS",
}

_GYM_MATCH_TYPES = {"GY4", "GY6", "GY8"}

_SCENARIO_CONFIGS: dict[str, dict[str, float]] = {
    "low": {"yield": 0.9, "within_school_dispersion": 0.6},
    "central": {"yield": 0.65, "within_school_dispersion": 0.75},
    "high": {"yield": 0.4, "within_school_dispersion": 0.9},
}
_SCENARIO_P_MIN = 0.02
_SCENARIO_P_MAX = 1.0


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scenario_uplift(*, p: float, dispersion: float) -> float:
    if p >= 1.0:
        return 0.0
    z_tail = NormalDist().inv_cdf(1.0 - p)
    return float(dispersion * _normal_pdf(z_tail) / p)


def _build_capacity_proxy(mz_pref: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "school_key",
        "match_school_type",
        "graduation_year",
        "capacity_throughput_proxy_candidates",
        "capacity_throughput_proxy_component_source",
        "capacity_throughput_proxy_variant_source",
        "capacity_throughput_proxy_source_year",
        "capacity_throughput_proxy_source_id",
        "capacity_throughput_proxy_method",
    ]
    if mz_pref.empty:
        return pd.DataFrame(columns=cols)

    proxy_rows: list[dict[str, Any]] = []
    cands = mz_pref[_numeric_series(mz_pref, "candidates").notna()].copy()
    if cands.empty:
        return pd.DataFrame(columns=cols)
    cands["candidates"] = pd.to_numeric(cands["candidates"], errors="coerce")
    cands = cands[cands["component"].isin(["CJ", "M", "AJ"])].copy()
    cands = cands[cands["candidates"] > 0].copy()
    if cands.empty:
        return pd.DataFrame(columns=cols)

    for (school_key, school_type, year), g in cands.groupby(["school_key", "match_school_type", "year"], dropna=False):
        yv = pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0]
        if pd.isna(yv):
            continue
        g2 = g.dropna(subset=["candidates"]).copy()
        if g2.empty:
            continue
        idx = g2["candidates"].astype(float).idxmax()
        row = g2.loc[idx]
        proxy_rows.append(
            {
                "school_key": school_key,
                "match_school_type": school_type,
                "graduation_year": int(yv),
                "capacity_throughput_proxy_candidates": float(row.get("candidates")),
                "capacity_throughput_proxy_component_source": row.get("component"),
                "capacity_throughput_proxy_variant_source": row.get("variant"),
                "capacity_throughput_proxy_source_year": int(yv),
                "capacity_throughput_proxy_source_id": row.get("source_id"),
                "capacity_throughput_proxy_method": "max_mz_candidates_across_CJ_M_AJ_same_school_type_graduation_year",
            }
        )
    if not proxy_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(proxy_rows)[cols]


def _build_scenario_selectivity_and_outcomes(jpz: pd.DataFrame, mz_pref: pd.DataFrame) -> pd.DataFrame:
    if jpz.empty or mz_pref.empty:
        return pd.DataFrame(
            columns=[
                "entry_year",
                "graduation_year",
                "school_key",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "jpz_cj_mean_percentile",
                "jpz_m_mean_percentile",
                "jpz_cj_sat",
                "jpz_m_sat",
                "jpz_test_takers_cj_m_mean",
                "jpz_test_takers_cj_m_mean_method",
                "capacity_throughput_proxy_candidates",
                "capacity_throughput_proxy_component_source",
                "capacity_throughput_proxy_variant_source",
                "capacity_throughput_proxy_source_year",
                "capacity_throughput_proxy_source_id",
                "capacity_throughput_proxy_method",
                "synthetic_input_cj_m_z_mean",
                "synthetic_offer_fraction_denominator_source",
                "synthetic_offer_fraction_central",
                "synthetic_offer_fraction_low",
                "synthetic_offer_fraction_high",
                "synthetic_uplift_central",
                "synthetic_uplift_low",
                "synthetic_uplift_high",
                "synthetic_admitted_intake_selectivity_latent",
                "synthetic_admitted_intake_selectivity_latent_low",
                "synthetic_admitted_intake_selectivity_latent_high",
                "synthetic_admitted_intake_selectivity_percentile",
                "synthetic_admitted_intake_selectivity_percentile_n_schools",
                "outcome_component",
                "outcome_mz_mean_score_pct",
                "outcome_mz_school_mean_percentile",
                "outcome_mz_candidates",
                "outcome_score_valid",
                "outcome_candidates_valid",
                "scenario_selectivity_valid",
                "plot_eligible",
                "plot_exclusion_reason",
                "outcome_variant",
                "outcome_source_id",
            ]
        )

    jpz2 = jpz[jpz["component"].isin(["CJ", "M"])].copy()
    jpz2["entry_year"] = pd.to_numeric(jpz2.get("entry_year"), errors="coerce")
    jpz2["programme_duration_years"] = pd.to_numeric(jpz2.get("programme_duration_years"), errors="coerce")
    jpz2["graduation_year"] = jpz2["entry_year"] + jpz2["programme_duration_years"]
    jpz2 = jpz2[jpz2["entry_year"].notna() & jpz2["graduation_year"].notna()].copy()
    if jpz2.empty:
        return pd.DataFrame()

    keys = [
        "school_key",
        "match_school_type",
        "entry_year",
        "graduation_year",
    ]
    metadata_keys = [
        "redizo",
        "school_name_raw",
        "address_raw",
        "city",
        "postcode",
        "programme_taxonomy",
        "programme_identity",
        "classification_quality",
    ]

    def _first_non_null(series: pd.Series) -> object:
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else pd.NA

    meta = jpz2.groupby(keys, dropna=False).agg({c: _first_non_null for c in metadata_keys}).reset_index()
    agg = (
        jpz2.groupby(keys + ["component"], dropna=False)
        .agg(mean_percentile=("mean_percentile", "mean"), sat=("sat", "mean"), registered=("registered", "mean"))
        .reset_index()
    )
    wide = agg.set_index(keys + ["component"])[["mean_percentile", "sat", "registered"]].unstack("component").reset_index()
    wide.columns = [
        "_".join([str(x) for x in col if str(x) not in {"", "None"}]).strip("_") if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    wide = wide.merge(meta, on=keys, how="left")
    rename_map = {
        "mean_percentile_CJ": "jpz_cj_mean_percentile",
        "mean_percentile_M": "jpz_m_mean_percentile",
        "sat_CJ": "jpz_cj_sat",
        "sat_M": "jpz_m_sat",
        "registered_CJ": "jpz_cj_registered",
        "registered_M": "jpz_m_registered",
    }
    wide = wide.rename(columns=rename_map)

    for c in [
        "jpz_cj_mean_percentile",
        "jpz_m_mean_percentile",
        "jpz_cj_sat",
        "jpz_m_sat",
        "jpz_cj_registered",
        "jpz_m_registered",
    ]:
        wide[c] = _numeric_series(wide, c)

    cj_p = (wide["jpz_cj_mean_percentile"] / 100.0).clip(_SCENARIO_P_MIN, 0.98)
    m_p = (wide["jpz_m_mean_percentile"] / 100.0).clip(_SCENARIO_P_MIN, 0.98)
    valid_input = wide["jpz_cj_mean_percentile"].notna() & wide["jpz_m_mean_percentile"].notna()
    wide["synthetic_input_cj_m_z_mean"] = pd.NA
    if valid_input.any():
        z_vals = (cj_p.map(NormalDist().inv_cdf) + m_p.map(NormalDist().inv_cdf)) / 2.0
        wide.loc[valid_input, "synthetic_input_cj_m_z_mean"] = z_vals[valid_input]

    proxy = _build_capacity_proxy(mz_pref)
    wide["graduation_year"] = _numeric_series(wide, "graduation_year").astype("Int64")
    if not proxy.empty:
        proxy["graduation_year"] = _numeric_series(proxy, "graduation_year").astype("Int64")
    wide = wide.merge(proxy, on=["school_key", "match_school_type", "graduation_year"], how="left")
    wide["capacity_throughput_proxy_candidates"] = _numeric_series(wide, "capacity_throughput_proxy_candidates")

    wide["jpz_test_takers_cj_m_mean"] = pd.NA
    wide["jpz_test_takers_cj_m_mean_method"] = pd.NA
    cj_test_takers = _numeric_series(wide, "jpz_cj_sat").where(_numeric_series(wide, "jpz_cj_sat").gt(0), _numeric_series(wide, "jpz_cj_registered"))
    m_test_takers = _numeric_series(wide, "jpz_m_sat").where(_numeric_series(wide, "jpz_m_sat").gt(0), _numeric_series(wide, "jpz_m_registered"))
    test_takers_mask = cj_test_takers.notna() & m_test_takers.notna() & (cj_test_takers > 0) & (m_test_takers > 0)
    if test_takers_mask.any():
        wide.loc[test_takers_mask, "jpz_test_takers_cj_m_mean"] = (cj_test_takers[test_takers_mask] + m_test_takers[test_takers_mask]) / 2.0
        wide.loc[test_takers_mask, "jpz_test_takers_cj_m_mean_method"] = "jpz_sat_or_registered_mean"
    missing_test_takers_mask = (
        wide["jpz_test_takers_cj_m_mean"].isna()
        & _numeric_series(wide, "capacity_throughput_proxy_candidates").gt(0)
        & wide["match_school_type"].isin(["GY6", "GY8"])
    )
    if missing_test_takers_mask.any():
        wide.loc[missing_test_takers_mask, "jpz_test_takers_cj_m_mean"] = wide.loc[missing_test_takers_mask, "capacity_throughput_proxy_candidates"] / float(_SCENARIO_CONFIGS["central"]["yield"])
        wide.loc[missing_test_takers_mask, "jpz_test_takers_cj_m_mean_method"] = "proxy_div_yield_central"

    for scen_name, scen in _SCENARIO_CONFIGS.items():
        yld = float(scen["yield"])
        disp = float(scen["within_school_dispersion"])
        offer_col = f"synthetic_offer_fraction_{scen_name}"
        uplift_col = f"synthetic_uplift_{scen_name}"
        latent_col = "synthetic_admitted_intake_selectivity_latent" if scen_name == "central" else f"synthetic_admitted_intake_selectivity_latent_{scen_name}"
        wide[offer_col] = pd.NA
        wide[uplift_col] = pd.NA
        wide[latent_col] = pd.NA
        if "synthetic_offer_fraction_denominator_source" not in wide.columns:
            wide["synthetic_offer_fraction_denominator_source"] = pd.NA
        ok = (
            _numeric_series(wide, "synthetic_input_cj_m_z_mean").notna()
            & _numeric_series(wide, "jpz_test_takers_cj_m_mean").notna()
            & _numeric_series(wide, "capacity_throughput_proxy_candidates").notna()
            & (_numeric_series(wide, "jpz_test_takers_cj_m_mean") > 0)
            & (_numeric_series(wide, "capacity_throughput_proxy_candidates") > 0)
        )
        if ok.any():
            a = _numeric_series(wide.loc[ok], "jpz_test_takers_cj_m_mean")
            c = _numeric_series(wide.loc[ok], "capacity_throughput_proxy_candidates")
            p_raw = c / (a * yld)
            p = p_raw.clip(_SCENARIO_P_MIN, _SCENARIO_P_MAX)
            u = p.map(lambda pp: _scenario_uplift(p=float(pp), dispersion=disp))
            z0 = _numeric_series(wide.loc[ok], "synthetic_input_cj_m_z_mean")
            wide.loc[ok, offer_col] = p
            wide.loc[ok, uplift_col] = u
            wide.loc[ok, latent_col] = z0 + u
            wide.loc[ok, "synthetic_offer_fraction_denominator_source"] = (
                wide.loc[ok, "jpz_test_takers_cj_m_mean_method"]
                .astype("string")
                .replace(
                    {
                        "jpz_sat_or_registered_mean": "observed_jpz_test_takers_cj_m_mean",
                        "proxy_div_yield_central": "proxy_derived_from_capacity_proxy_div_yield_central",
                    }
                )
            )

    wide["synthetic_admitted_intake_selectivity_percentile"] = pd.NA
    wide["synthetic_admitted_intake_selectivity_percentile_n_schools"] = pd.NA
    wide["scenario_selectivity_valid"] = False
    for (_, _), g in wide.groupby(["entry_year", "match_school_type"], dropna=False):
        numeric = pd.to_numeric(g["synthetic_admitted_intake_selectivity_latent"], errors="coerce")
        pct, n = _equal_weight_percentile_rank(numeric)
        wide.loc[g.index, "synthetic_admitted_intake_selectivity_percentile"] = pct
        wide.loc[g.index, "synthetic_admitted_intake_selectivity_percentile_n_schools"] = n
        wide.loc[g.index, "scenario_selectivity_valid"] = numeric.notna() & pct.notna()

    outcomes = mz_pref[mz_pref["component"].isin(["CJ", "M", "AJ"])].copy()
    outcomes = outcomes.rename(
        columns={
            "year": "graduation_year",
            "component": "outcome_component",
            "mean_score": "outcome_mz_mean_score_pct",
            "candidates": "outcome_mz_candidates",
            "variant": "outcome_variant",
            "source_id": "outcome_source_id",
            "mz_school_mean_percentile": "outcome_mz_school_mean_percentile",
        }
    )
    for c in ["outcome_mz_mean_score_pct", "outcome_mz_candidates", "outcome_variant", "outcome_source_id", "outcome_mz_school_mean_percentile"]:
        if c not in outcomes.columns:
            outcomes[c] = pd.NA
    outcomes["outcome_score_valid"] = _numeric_series(outcomes, "outcome_mz_mean_score_pct").notna()
    outcomes["outcome_candidates_valid"] = _numeric_series(outcomes, "outcome_mz_candidates").gt(0).fillna(False)
    outcomes = outcomes.reindex(
        columns=[
            "school_key",
            "match_school_type",
            "graduation_year",
            "outcome_component",
            "outcome_mz_mean_score_pct",
            "outcome_mz_school_mean_percentile",
            "outcome_mz_candidates",
            "outcome_variant",
            "outcome_source_id",
        ]
    ).copy()
    out = wide.merge(outcomes, on=["school_key", "match_school_type", "graduation_year"], how="inner")
    out = out.rename(columns={"match_school_type": "school_type"})
    out["outcome_score_valid"] = _numeric_series(out, "outcome_mz_mean_score_pct").notna()
    out["outcome_candidates_valid"] = _numeric_series(out, "outcome_mz_candidates").gt(0).fillna(False)
    out["scenario_selectivity_valid"] = _numeric_series(out, "synthetic_admitted_intake_selectivity_percentile").notna()
    if out["scenario_selectivity_valid"].isna().any():
        out.loc[out["scenario_selectivity_valid"].isna() & out["school_type"].isin(["GY6", "GY8"]), "scenario_selectivity_valid"] = False
        out.loc[out["scenario_selectivity_valid"].isna() & out["school_type"].eq("GY4"), "scenario_selectivity_valid"] = False
    out["plot_eligible"] = out["scenario_selectivity_valid"] & out["outcome_score_valid"] & out["outcome_candidates_valid"]
    out["plot_exclusion_reason"] = pd.NA
    out.loc[~out["scenario_selectivity_valid"], "plot_exclusion_reason"] = "invalid_selectivity"
    out.loc[out["scenario_selectivity_valid"] & ~out["outcome_candidates_valid"], "plot_exclusion_reason"] = "invalid_outcome_candidates"
    out.loc[out["scenario_selectivity_valid"] & out["outcome_candidates_valid"] & ~out["outcome_score_valid"], "plot_exclusion_reason"] = "invalid_outcome_score"
    out["entry_year"] = _numeric_series(out, "entry_year").astype("Int64")
    out["graduation_year"] = _numeric_series(out, "graduation_year").astype("Int64")
    return out


def _classify_programme_taxonomy(
    programme_group_raw: object,
    *,
    kkov_raw: object = None,
    grade: object = None,
) -> tuple[str, object, object, str, str]:
    raw_group = _safe_text(programme_group_raw)
    raw_kkov = _safe_text(kkov_raw)
    compact_group = re.sub(r"[^a-z0-9]+", "", _norm_text(raw_group)) if raw_group is not None else ""
    compact_kkov = re.sub(r"[^a-z0-9]+", "", _norm_text(raw_kkov)) if raw_kkov is not None else ""

    if compact_kkov.endswith("41"):
        return "GY4", 4, 9, "GY4", "authoritative_kkov_duration"
    if compact_kkov.endswith("61"):
        return "GY6", 6, 7, "GY6", "authoritative_kkov_duration"
    if compact_kkov.endswith("81"):
        return "GY8", 8, 5, "GY8", "authoritative_kkov_duration"

    if compact_group in _PROGRAMME_GROUP16_TAXONOMY_MAP:
        taxonomy = _PROGRAMME_GROUP16_TAXONOMY_MAP[compact_group]
        if taxonomy == "GY4":
            return taxonomy, 4, 9, "GY4", "authoritative_programme_group16"
        if taxonomy == "GY6":
            return taxonomy, 6, 7, "GY6", "authoritative_programme_group16"
        if taxonomy == "GY8":
            return taxonomy, 8, 5, "GY8", "authoritative_programme_group16"
        if taxonomy == "TOTAL_AGGREGATE":
            return taxonomy, pd.NA, pd.NA, "TOTAL_AGGREGATE", "authoritative_programme_group16"
        if taxonomy in _SOS_MATCH_TYPES:
            return taxonomy, 4, 9, taxonomy, "authoritative_programme_group16"
        return taxonomy, pd.NA, pd.NA, taxonomy, "authoritative_programme_group16"

    if compact_group in {"7941k81", "7941k61", "7941k41"}:
        mapping = {
            "7941k41": ("GY4", 4, 9),
            "7941k61": ("GY6", 6, 7),
            "7941k81": ("GY8", 8, 5),
        }
        taxonomy, duration, entrant_grade = mapping[compact_group]
        return taxonomy, duration, entrant_grade, taxonomy, "authoritative_historic_group"

    g = pd.to_numeric(pd.Series([grade]), errors="coerce").iloc[0]
    if compact_group == "7941k" and pd.notna(g):
        gi = int(g)
        if gi == 5:
            return "GY8", 8, 5, "GY8", "authoritative_historic_group_grade"
        if gi == 7:
            return "GY6", 6, 7, "GY6", "authoritative_historic_group_grade"
        if gi == 9:
            return "GY4", 4, 9, "GY4", "authoritative_historic_group_grade"

    return "UNKNOWN", pd.NA, pd.NA, "UNKNOWN", "unknown"


def _attach_school_type_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["school_type"] = pd.Series(dtype="string")
        out["programme_taxonomy"] = pd.Series(dtype="string")
        out["programme_identity"] = pd.Series(dtype="string")
        out["programme_duration_years"] = pd.Series(dtype="Int64")
        out["entrant_grade"] = pd.Series(dtype="Int64")
        out["classification_quality"] = pd.Series(dtype="string")
        return out

    out = df.copy()
    program = out.get("programme_group_16_raw", out.get("programme_group_raw", pd.Series([pd.NA] * len(out), index=out.index)))
    kkov = out.get("kkov_raw", pd.Series([pd.NA] * len(out), index=out.index))
    grade = out.get("grade", pd.Series([pd.NA] * len(out), index=out.index))
    classified = [
        _classify_programme_taxonomy(p, kkov_raw=k, grade=g)
        for p, k, g in zip(program.tolist(), kkov.tolist(), grade.tolist())
    ]
    out["school_type"] = pd.Series([x[0] for x in classified], index=out.index, dtype="string")
    out["programme_taxonomy"] = pd.Series([x[0] for x in classified], index=out.index, dtype="string")
    out["programme_duration_years"] = pd.Series([x[1] for x in classified], index=out.index, dtype="Int64")
    out["entrant_grade"] = pd.Series([x[2] for x in classified], index=out.index, dtype="Int64")
    out["programme_identity"] = pd.Series([x[3] for x in classified], index=out.index, dtype="string")
    out["classification_quality"] = pd.Series([x[4] for x in classified], index=out.index, dtype="string")
    return out


def _choose_best_value(group: pd.DataFrame, field: str, *, prefer_non_null: bool = True) -> pd.Series:
    if field not in group.columns:
        return pd.Series([pd.NA] * len(group), index=group.index)
    values = group[field].astype("string")
    if not prefer_non_null:
        return values
    non_empty = values.notna() & values.str.strip().ne("")
    return values.where(non_empty)


def _best_non_null(group: pd.DataFrame, field: str) -> tuple[object, str | None]:
    if field not in group.columns:
        return pd.NA, None
    rows = group[[field, "source_id"]].copy()
    rows["_text"] = rows[field].map(_safe_text)
    rows = rows[rows["_text"].notna()].copy()
    if rows.empty:
        return pd.NA, None
    freq = rows["_text"].value_counts(dropna=True).to_dict()
    rows["_freq"] = rows["_text"].map(freq).fillna(0).astype(int)
    rows["_len"] = rows["_text"].astype(str).str.len()
    rows["_source_rank"] = rows["source_id"].map(lambda s: {"jpz": 0, "mz": 1}.get(str(s).split("_")[0], 9))
    chosen = rows.sort_values(["_freq", "_len", "_source_rank", "source_id", "_text"], ascending=[False, False, True, True, True]).iloc[0]
    return chosen["_text"], str(chosen["source_id"])


def _linear_slope(years: pd.Series, values: pd.Series) -> float | None:
    y = pd.to_numeric(values, errors="coerce")
    x = pd.to_numeric(years, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return None
    xv = x[mask].astype(float).to_numpy()
    yv = y[mask].astype(float).to_numpy()
    return float(np.polyfit(xv, yv, 1)[0])


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def _redizo_from_school_key(value: object) -> str | None:
    text = _safe_text(value)
    if text is None:
        return None
    m = re.match(r"^redizo:(\d{6,10})$", text)
    return m.group(1) if m else None


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


def _attach_mz_school_mean_percentile(mz_pref: pd.DataFrame) -> pd.DataFrame:
    if mz_pref.empty:
        out = mz_pref.copy()
        out["mz_school_mean_percentile"] = pd.Series(dtype="Float64")
        out["mz_school_mean_percentile_method"] = pd.Series(dtype="string")
        out["mz_school_mean_percentile_reference"] = pd.Series(dtype="string")
        out["mz_school_mean_percentile_n_schools"] = pd.Series(dtype="Int64")
        out["mz_school_mean_percentile_valid"] = pd.Series(dtype="boolean")
        return out

    out = mz_pref.copy()
    out["mz_school_mean_percentile"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    out["mz_school_mean_percentile_method"] = "equal_weight_school_mean_percentile_rank"
    out["mz_school_mean_percentile_reference"] = out.apply(
        lambda r: f"eligible schools in graduation_year={r['year']} component={r['component']}",
        axis=1,
    )
    out["mz_school_mean_percentile_n_schools"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["mz_school_mean_percentile_valid"] = False

    valid_mask = _numeric_series(out, "candidates").gt(0) & _numeric_series(out, "mean_score").notna()
    for (_, _), g in out[valid_mask].groupby(["year", "component"], dropna=False):
        pct, n = _equal_weight_percentile_rank(g["mean_score"])
        out.loc[g.index, "mz_school_mean_percentile"] = pct
        out.loc[g.index, "mz_school_mean_percentile_n_schools"] = n
        out.loc[g.index, "mz_school_mean_percentile_valid"] = True
    return out


def _compute_mz_participation_rate_vs_cj(mz_pref: pd.DataFrame) -> pd.DataFrame:
    """MZ participation rate (relative to CJ) per school_key + school_type + graduation_year.

    Defined as component MZ candidates / CJ MZ candidates for the same
    (school_key, match_school_type, year) group. `mz_pref` is expected to already be
    deduplicated by variant preference (jap > j), matching the pattern used elsewhere
    (see `_attach_mz_school_mean_percentile` callers). Candidates are summed across any
    remaining duplicate rows sharing the same key before the ratio is taken. CJ rows
    therefore evaluate to ~1.0 (a data-quality signal), while M/AJ rows are typically
    lower. Division by a missing or zero CJ candidate count yields a null rate.
    """
    cols = ["school_key", "match_school_type", "year", "component", "mz_participation_rate_vs_cj"]
    if mz_pref.empty:
        return pd.DataFrame(columns=cols)

    df = mz_pref.copy()
    df["candidates"] = pd.to_numeric(df.get("candidates"), errors="coerce")
    grp = (
        df.groupby(["school_key", "match_school_type", "year", "component"], dropna=False)["candidates"]
        .sum(min_count=1)
        .reset_index()
    )
    cj = grp[grp["component"].eq("CJ")][["school_key", "match_school_type", "year", "candidates"]].rename(
        columns={"candidates": "cj_candidates"}
    )
    out = grp.merge(cj, on=["school_key", "match_school_type", "year"], how="left")
    out["mz_participation_rate_vs_cj"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    valid = out["cj_candidates"].notna() & out["cj_candidates"].gt(0) & out["candidates"].notna()
    out.loc[valid, "mz_participation_rate_vs_cj"] = (
        out.loc[valid, "candidates"] / out.loc[valid, "cj_candidates"]
    )
    return out[cols]


@dataclass(frozen=True)
class PlannedSource:
    source_id: str
    dataset: str
    year: int
    kind: str
    url: str | None
    required: bool
    parser_profile: str
    unavailable_reason: str | None = None


def build_source_plan(year_start: int = 2016, year_end: int = 2026) -> list[PlannedSource]:
    out: list[PlannedSource] = []

    for year in range(max(2016, year_start), min(2023, year_end) + 1):
        if year == 2016:
            out.append(
                PlannedSource(
                    source_id="jpz_2016_historic_vysledky",
                    dataset="jpz",
                    year=2016,
                    kind="historic_vysledky",
                    url=None,
                    required=False,
                    parser_profile="jpz_historic_2017_2023",
                    unavailable_reason="Official JPZ aggregated school-level workbook is publicly available from 2017 onward; 2016 unavailable.",
                )
            )
            continue
        out.append(
            PlannedSource(
                source_id=f"jpz_{year}_historic_vysledky",
                dataset="jpz",
                year=year,
                kind="historic_vysledky",
                url=JPZ_HISTORIC_URL.format(year=year),
                required=True,
                parser_profile="jpz_historic_2017_2023",
            )
        )

    for year in range(max(2024, year_start), min(2026, year_end) + 1):
        for kind in ("kapacity", "prihlasky", "vysledky"):
            out.append(
                PlannedSource(
                    source_id=f"jpz_{year}_kolo1_{kind}",
                    dataset="jpz",
                    year=year,
                    kind=f"modern_{kind}",
                    url=JPZ_MODERN_URL.format(year=year, kind=kind),
                    required=kind in {"kapacity", "prihlasky"},
                    parser_profile="jpz_modern_kolo1_2024_2026",
                )
            )

    for year in range(max(2015, year_start), min(2026, year_end) + 1):
        for variant in ("j", "jap"):
            out.append(
                PlannedSource(
                    source_id=f"mz_{year}_{variant}",
                    dataset="mz",
                    year=year,
                    kind=f"{variant}",
                    url=MZ_URL.format(year=year, variant=variant),
                    required=False,
                    parser_profile="mz_wide_components_2015_2026",
                )
            )

    return out


def _load_reusable_url_index(archive_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not archive_root.exists():
        return index
    for child in archive_root.iterdir():
        manifest = child / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("sources", []):
            if item.get("status") != "downloaded":
                continue
            url = item.get("url")
            rel = item.get("raw_file")
            if not url or not rel:
                continue
            p = child / rel
            if p.exists():
                index[str(url)] = p
    return index


def _materialize_local(source_file: Path, target_file: Path) -> None:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists():
        return
    try:
        target_file.hardlink_to(source_file)
    except Exception:
        shutil.copy2(source_file, target_file)


def create_archive(
    archive_root: str | Path,
    freeze_id: str | None = None,
    year_start: int = 2016,
    year_end: int = 2026,
    refresh: bool = False,
    timeout: int = 60,
    retries: int = 3,
    source_override: str | Path | None = None,
) -> Path:
    root = ensure_dir(archive_root)
    fid = freeze_id or _freeze_id()
    archive_dir = root / fid
    if archive_dir.exists() and any(archive_dir.iterdir()):
        raise FileExistsError(f"Archive freeze already exists and is non-empty: {archive_dir}")
    raw_dir = ensure_dir(archive_dir / "raw")
    ensure_dir(archive_dir / "normalized")
    ensure_dir(archive_dir / "reports")

    override_map: dict[str, str] = {}
    if source_override:
        override_map = {str(k): str(v) for k, v in load_json(source_override).items()}
    reusable = _load_reusable_url_index(root)

    manifest: dict[str, Any] = {
        "freeze_id": fid,
        "created_at": _now_iso(),
        "archive_dir": str(archive_dir),
        "network_policy": "Network fetch is allowed only in archive creation.",
        "sources": [],
    }

    for src in build_source_plan(year_start=year_start, year_end=year_end):
        item: dict[str, Any] = {
            "source_id": src.source_id,
            "dataset": src.dataset,
            "year": src.year,
            "kind": src.kind,
            "url": src.url,
            "required": src.required,
            "parser_profile": src.parser_profile,
            "timestamp": _now_iso(),
        }
        if src.url is None:
            item["status"] = "unavailable"
            item["reason"] = src.unavailable_reason
            manifest["sources"].append(item)
            continue

        target_name = f"{src.source_id}.xlsx"
        target_file = raw_dir / target_name
        override_path = override_map.get(src.source_id) or override_map.get(src.url)

        try:
            if override_path:
                source_file = Path(override_path)
                if not source_file.exists():
                    raise FileNotFoundError(f"Override file not found: {source_file}")
                _materialize_local(source_file, target_file)
                item["status"] = "downloaded"
                item["fetch_mode"] = "override_local"
                item["http_status"] = None
                item["headers"] = {}
            elif (not refresh) and src.url in reusable:
                _materialize_local(reusable[src.url], target_file)
                item["status"] = "downloaded"
                item["fetch_mode"] = "reused_existing_archive"
                item["http_status"] = None
                item["headers"] = {}
            else:
                meta = download_url(src.url, target_file, timeout=timeout, retries=retries, require_excel=True)
                item["status"] = "downloaded"
                item["fetch_mode"] = "network"
                item["http_status"] = int(meta.get("status", 0))
                item["headers"] = {
                    "content_type": meta.get("content_type", ""),
                    "etag": meta.get("etag", ""),
                    "last_modified": meta.get("last_modified", ""),
                }
            item["raw_file"] = str(Path("raw") / target_name)
            item["bytes"] = int(target_file.stat().st_size)
            item["sha256"] = sha256_of_file(target_file)
        except Exception as exc:
            item["status"] = "unavailable"
            item["reason"] = str(exc)
            if src.required:
                item["required_missing"] = True
        manifest["sources"].append(item)

    manifest_path = archive_dir / "manifest.json"
    dump_json(manifest_path, manifest)
    return archive_dir


def _open_excel_multiheader(path: Path) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    best: tuple[int, int, pd.DataFrame, int] | None = None
    for hdr in (0, 1, 2):
        try:
            df = pd.read_excel(path, header=[hdr, hdr + 1])
            if len(df.columns) <= 3:
                continue
            if len(df) < 2:
                continue
            flat_cols = _flatten_cols(df.columns)
            has_identity = _pick_col(flat_cols, "redizo", "red_izo", "izo") and _pick_col(flat_cols, "nazev")
            if not has_identity:
                continue
            lvl1_tokens = [
                _norm_text(c[1])
                for c in df.columns
                if isinstance(c, tuple) and len(c) >= 2 and _safe_text(c[1]) is not None
            ]
            header_hits = sum(
                1
                for token in lvl1_tokens
                if any(k in token for k in ("prihlas", "konali", "percentil", "redizo", "rocnik", "nazev", "adresa", "obor", "skola"))
            )
            data_like_hits = sum(
                1
                for token in lvl1_tokens
                if re.search(r"\d", token) or any(k in token for k in ("gymnaz", "ulice", "mesto", "brno", "praha", "ostrava"))
            )
            if header_hits < 3 or data_like_hits > max(2, len(lvl1_tokens) // 3):
                continue
            metric_hits = 0
            for col in flat_cols:
                comp = _extract_component_from_text(col)
                metric = _metric_from_text(col)
                if comp in {"CJ", "M"} and metric in {"mean_percentile", "registered", "sat"}:
                    metric_hits += 1
            score = (metric_hits, -hdr)
            if best is None or score > (best[0], best[1]):
                best = (metric_hits, -hdr, df, hdr)
        except Exception:
            continue
    if best is not None:
        return best[2], best[3]
    return None, None


def _flatten_cols(cols: pd.Index) -> list[str]:
    def _clean_token(token: object) -> str:
        if token is None:
            return ""
        try:
            if pd.isna(token):
                return ""
        except Exception:
            pass
        text = str(token).strip()
        if not text:
            return ""
        if text.lower().startswith("unnamed:"):
            return ""
        return text

    out: list[str] = []
    last_lvl0 = ""
    for col in cols:
        if isinstance(col, tuple):
            c0 = _clean_token(col[0] if len(col) >= 1 else "")
            c1 = _clean_token(col[1] if len(col) >= 2 else "")
            if not c0:
                c0 = last_lvl0
            else:
                last_lvl0 = c0
            out.append(f"{c0} | {c1}".strip(" |"))
        else:
            out.append(str(col))
    return out


def _pick_col(columns: list[str], *hints: str) -> str | None:
    nmap = [(_norm_text(c), c) for c in columns]
    for hint in hints:
        h = _norm_text(hint)
        best: tuple[int, int, str] | None = None
        for ncol, raw in nmap:
            if h not in ncol:
                continue
            if ncol == h:
                score = (0, len(ncol), raw)
            elif ncol.startswith(h):
                score = (1, len(ncol), raw)
            else:
                score = (2, len(ncol), raw)
            if best is None or score < best:
                best = score
        if best is not None:
            return best[2]
    return None


def _extract_component_from_text(text: str) -> str | None:
    n = _norm_text(text)
    if any(x in n for x in ("cesk", "cjl", "čj", "cj")):
        return "CJ"
    if "matemat" in n or re.search(r"(^|\W)(m|mat)(\W|$)", n):
        return "M"
    if "anglic" in n or "anglict" in n or re.search(r"(^|\W)aj(\W|$)", n):
        return "AJ"
    if "smo16" in n:
        return None
    if any(x in n for x in ("celkem", "total")):
        return "TOTAL"
    return None


def _metric_from_text(text: str) -> str | None:
    n = _norm_text(text)
    if "sm" in n and "odch" in n:
        return "sd"
    if "percentil" in n:
        return "mean_percentile"
    if "konali" in n or "konav" in n:
        if "nekonali" in n:
            return None
        return "sat"
    if "prihlas" in n:
        return "registered"
    if "prijat" in n:
        return "admitted"
    if "% skor" in n or "prumerny %" in n or "průměrný %" in text.lower():
        return "mean_score"
    if "podil" in n and "uspes" in n:
        return "pass_rate"
    return None


def parse_jpz_historic_components(path: Path, entry_year: int, source_id: str, _force_single_header: bool = False) -> pd.DataFrame:
    df2, hdr = _open_excel_multiheader(path) if not _force_single_header else (None, None)
    if df2 is not None:
        flat_cols = _flatten_cols(df2.columns)
        df = df2.copy()
        df.columns = flat_cols
    else:
        chosen: pd.DataFrame | None = None
        chosen_hdr = 0
        for h in (1, 0, 2, 3):
            try:
                d = pd.read_excel(path, header=h)
            except Exception:
                continue
            cols = [str(c) for c in d.columns]
            if _pick_col(cols, "redizo", "red_izo") and _pick_col(cols, "nazev", "školy", "skoly"):
                chosen = d
                chosen_hdr = h
                break
        if chosen is None:
            chosen = pd.read_excel(path, header=0)
        df = chosen
        hdr = chosen_hdr

    cols = [str(c) for c in df.columns]
    if not any(_metric_from_text(c) in {"mean_percentile", "registered", "sat"} for c in cols):
        try:
            alt = pd.read_excel(path, header=[hdr, hdr + 1])
            flat_alt = _flatten_cols(alt.columns)
            if _pick_col(flat_alt, "redizo", "red_izo") and _pick_col(flat_alt, "nazev") and sum(
                1 for c in flat_alt if _metric_from_text(c) in {"mean_percentile", "registered", "sat"}
            ) >= 2:
                df = alt.copy()
                df.columns = flat_alt
                cols = flat_alt
        except Exception:
            pass
    school_id_col = _pick_col(cols, "redizo", "red_izo", "red izo", "redizo /", "izo")
    school_name_col = _pick_col(cols, "nazev skoly", "název školy", "nazev")
    address_col = _pick_col(cols, "adresa", "adresa skoly")
    grade_col = _pick_col(cols, "rocnik", "ročník")
    group_col = _pick_col(cols, "oborova skupina", "oborová skupina", "smo16", "gy8", "kkov")
    kkov_col = _pick_col(cols, "kkov", "kod oboru", "obor kod")
    prog_name_col = _pick_col(cols, "nazev oboru", "obor", "oborova skupina", "oborová skupina")
    prog_focus_col = _pick_col(cols, "zamereni", "zaměření", "specializace")
    if group_col == school_id_col:
        group_candidates = [
            c
            for c in cols
            if c != school_id_col and any(h in _norm_text(c) for h in ("oborova skupina", "oborová skupina", "smo16", "kkov", "obor"))
        ]
        if group_candidates:
            group_col = group_candidates[0]

    if school_id_col is None or school_name_col is None:
        raise ValueError(f"Could not parse historic JPZ identity columns from {path}")

    id_series = df[school_id_col].map(_digits)
    if df2 is not None and int(id_series.notna().sum()) == 0:
        return parse_jpz_historic_components(path, entry_year, source_id, _force_single_header=True)
    name_series = df[school_name_col].astype("string").str.strip()
    addr_series = df[address_col].astype("string").str.strip() if address_col else pd.Series([pd.NA] * len(df), index=df.index)
    grade_series = pd.to_numeric(df[grade_col], errors="coerce") if grade_col else pd.Series([pd.NA] * len(df), index=df.index)
    group_series = df[group_col].astype("string").str.strip() if group_col else pd.Series([pd.NA] * len(df), index=df.index)
    kkov_series = df[kkov_col].astype("string").str.strip() if kkov_col else pd.Series([pd.NA] * len(df), index=df.index)
    prog_name_series = df[prog_name_col].astype("string").str.strip() if prog_name_col else pd.Series([pd.NA] * len(df), index=df.index)
    prog_focus_series = df[prog_focus_col].astype("string").str.strip() if prog_focus_col else pd.Series([pd.NA] * len(df), index=df.index)

    base_mask = id_series.notna()

    records: list[dict[str, Any]] = []
    metric_cols: dict[tuple[str, str], str] = {}
    percentile_columns: list[tuple[str, str]] = []
    for col in cols:
        comp = _extract_component_from_text(col)
        metric = _metric_from_text(col)
        if comp in {"CJ", "M"} and metric in {"mean_percentile", "sd", "registered", "sat"}:
            metric_cols[(comp, metric)] = col
            if metric == "mean_percentile":
                percentile_columns.append((comp, col))

    if len(percentile_columns) < 2:
        unnamed = [c for c in cols if "percentil" in _norm_text(c)]
        if len(unnamed) >= 2:
            percentile_columns = [("CJ", unnamed[0]), ("M", unnamed[1])]

    for idx, row in df[base_mask].iterrows():
        redizo = id_series.loc[idx]
        sname = name_series.loc[idx]
        saddr = _safe_text(addr_series.loc[idx])
        city, postcode = _city_postcode(saddr)
        grade = grade_series.loc[idx]
        pgroup = group_series.loc[idx]
        programme_taxonomy, duration_years, entrant_grade, programme_identity, classification_quality = _classify_programme_taxonomy(
            pgroup,
            kkov_raw=kkov_series.loc[idx],
            grade=grade,
        )
        by_comp: dict[str, float] = {}
        for comp in ("CJ", "M"):
            rec = {
                "entry_year": entry_year,
                "redizo": redizo,
                "school_name_raw": sname,
                "address_raw": saddr,
                "city": city,
                "postcode": postcode,
                "address_source_id": source_id if saddr is not None else pd.NA,
                "city_source_id": source_id if city is not None else pd.NA,
                "programme_group_raw": pgroup,
                "programme_group_16_raw": pgroup,
                "kkov_raw": kkov_series.loc[idx],
                "programme_name_raw": prog_name_series.loc[idx],
                "programme_focus_raw": prog_focus_series.loc[idx],
                "grade": grade,
                "school_type": programme_taxonomy,
                "programme_taxonomy": programme_taxonomy,
                "programme_identity": programme_identity,
                "programme_duration_years": duration_years,
                "entrant_grade": entrant_grade,
                "classification_quality": classification_quality,
                "component": comp,
                "mean_percentile": pd.NA,
                "sd": pd.NA,
                "registered": pd.NA,
                "sat": pd.NA,
                "admitted": pd.NA,
                "metric_name": "mean_percentile",
                "source_id": source_id,
                "source_row_number": int(idx) + int(hdr) + 3,
            }
            for metric in ("mean_percentile", "sd", "registered", "sat"):
                col = metric_cols.get((comp, metric))
                if col is None:
                    if metric == "mean_percentile":
                        fallback_col = next((c for cp, c in percentile_columns if cp == comp), None)
                        if fallback_col is not None:
                            rec[metric] = pd.to_numeric(pd.Series([row.get(fallback_col)]), errors="coerce").iloc[0]
                    continue
                rec[metric] = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(rec["mean_percentile"]):
                by_comp[comp] = float(rec["mean_percentile"])
            records.append(rec)

        observed = [by_comp.get("CJ"), by_comp.get("M")]
        observed = [x for x in observed if x is not None and not pd.isna(x)]
        if observed:
            records.append(
                {
                    "entry_year": entry_year,
                    "redizo": redizo,
                    "school_name_raw": sname,
                    "address_raw": saddr,
                    "city": city,
                    "postcode": postcode,
                    "address_source_id": source_id if saddr is not None else pd.NA,
                    "city_source_id": source_id if city is not None else pd.NA,
                    "programme_group_raw": pgroup,
                    "programme_group_16_raw": pgroup,
                    "kkov_raw": kkov_series.loc[idx],
                    "programme_name_raw": prog_name_series.loc[idx],
                    "programme_focus_raw": prog_focus_series.loc[idx],
                    "grade": grade,
                    "school_type": programme_taxonomy,
                    "programme_taxonomy": programme_taxonomy,
                    "programme_identity": programme_identity,
                    "programme_duration_years": duration_years,
                    "entrant_grade": entrant_grade,
                    "classification_quality": classification_quality,
                    "component": "CJ_M_EQUAL",
                    "mean_percentile": float(np.mean(observed)),
                    "sd": pd.NA,
                    "registered": pd.NA,
                    "sat": pd.NA,
                    "admitted": pd.NA,
                    "metric_name": "mean_percentile_cj_m_equal_weight",
                    "source_id": source_id,
                    "source_row_number": int(idx) + int(hdr) + 3,
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out[out["component"].isin(["CJ", "M", "CJ_M_EQUAL"])].reset_index(drop=True)


def parse_jpz_modern_triplet(
    app_path: Path,
    cap_path: Path,
    result_path: Path | None,
    year: int,
    app_source_id: str | None = None,
    cap_source_id: str | None = None,
    result_source_id: str | None = None,
) -> pd.DataFrame:
    app = pd.read_excel(app_path)
    cap = pd.read_excel(cap_path)
    app_cols = [str(c) for c in app.columns]
    cap_cols = [str(c) for c in cap.columns]

    app_id = _pick_col(app_cols, "redizo", "red_izo", "izo")
    cap_id = _pick_col(cap_cols, "redizo", "red_izo", "izo")
    app_name = _pick_col(app_cols, "nazev")
    cap_name = _pick_col(cap_cols, "nazev")
    app_addr = _pick_col(app_cols, "adresa")
    cap_addr = _pick_col(cap_cols, "adresa")
    app_group = _pick_col(app_cols, "smo16")
    cap_group = _pick_col(cap_cols, "smo16")
    app_kkov = _pick_col(app_cols, "kkov", "kod oboru", "obor kod")
    cap_kkov = _pick_col(cap_cols, "kkov", "kod oboru", "obor kod")
    app_prog_name = _pick_col(app_cols, "nazev oboru", "obor")
    cap_prog_name = _pick_col(cap_cols, "nazev oboru", "obor")
    app_prog_focus = _pick_col(app_cols, "zamereni", "zaměření", "specializace")
    cap_prog_focus = _pick_col(cap_cols, "zamereni", "zaměření", "specializace")
    app_val = _pick_col(app_cols, "prihlask")
    cap_val = _pick_col(cap_cols, "kapacit", "mista")

    if not all([app_id, cap_id, app_name, cap_name, app_val, cap_val]):
        raise ValueError(f"Could not parse modern JPZ triplet for year {year}")

    app_df = pd.DataFrame(
        {
            "redizo": app[app_id].map(_digits),
            "school_name_raw": app[app_name].astype("string").str.strip(),
            "address_raw": app[app_addr].astype("string").str.strip() if app_addr else pd.Series([pd.NA] * len(app), index=app.index),
            "programme_group_raw": app[app_group].astype("string").str.strip() if app_group else pd.Series([pd.NA] * len(app), index=app.index),
            "programme_group_16_raw": app[app_group].astype("string").str.strip() if app_group else pd.Series([pd.NA] * len(app), index=app.index),
            "kkov_raw": app[app_kkov].astype("string").str.strip() if app_kkov else pd.Series([pd.NA] * len(app), index=app.index),
            "programme_name_raw": app[app_prog_name].astype("string").str.strip() if app_prog_name else pd.Series([pd.NA] * len(app), index=app.index),
            "programme_focus_raw": app[app_prog_focus].astype("string").str.strip() if app_prog_focus else pd.Series([pd.NA] * len(app), index=app.index),
            "applications": pd.to_numeric(app[app_val], errors="coerce"),
        }
    )
    cap_df = pd.DataFrame(
        {
            "redizo": cap[cap_id].map(_digits),
            "school_name_raw": cap[cap_name].astype("string").str.strip(),
            "address_raw": cap[cap_addr].astype("string").str.strip() if cap_addr else pd.Series([pd.NA] * len(cap), index=cap.index),
            "programme_group_raw": cap[cap_group].astype("string").str.strip() if cap_group else pd.Series([pd.NA] * len(cap), index=cap.index),
            "programme_group_16_raw": cap[cap_group].astype("string").str.strip() if cap_group else pd.Series([pd.NA] * len(cap), index=cap.index),
            "kkov_raw": cap[cap_kkov].astype("string").str.strip() if cap_kkov else pd.Series([pd.NA] * len(cap), index=cap.index),
            "programme_name_raw": cap[cap_prog_name].astype("string").str.strip() if cap_prog_name else pd.Series([pd.NA] * len(cap), index=cap.index),
            "programme_focus_raw": cap[cap_prog_focus].astype("string").str.strip() if cap_prog_focus else pd.Series([pd.NA] * len(cap), index=cap.index),
            "capacity": pd.to_numeric(cap[cap_val], errors="coerce"),
        }
    )

    key = [
        "redizo",
        "school_name_raw",
        "address_raw",
        "programme_group_raw",
        "programme_group_16_raw",
        "kkov_raw",
        "programme_name_raw",
        "programme_focus_raw",
    ]
    merged = (
        app_df.groupby(key, dropna=False)["applications"].sum(min_count=1).reset_index()
        .merge(cap_df.groupby(key, dropna=False)["capacity"].sum(min_count=1).reset_index(), on=key, how="outer")
    )
    merged["entry_year"] = year
    merged["city"] = merged["address_raw"].map(lambda x: _city_postcode(x)[0])
    merged["postcode"] = merged["address_raw"].map(lambda x: _city_postcode(x)[1])
    merged["source_id"] = "+".join([x for x in [app_source_id, cap_source_id, result_source_id] if x]) or f"jpz_{year}_kolo1_triplet"
    merged["address_source_id"] = "+".join([x for x in [app_source_id, cap_source_id] if x]) or pd.NA
    merged["city_source_id"] = merged["address_source_id"]
    merged["actual_score_field"] = pd.NA
    merged["actual_score_value"] = pd.NA
    merged["actual_score_unit"] = pd.NA
    merged["grade"] = pd.NA

    if result_path and result_path.exists():
        res = pd.read_excel(result_path)
        rcols = [str(c) for c in res.columns]
        rid = _pick_col(rcols, "redizo", "red_izo", "izo")
        rval = _pick_col(rcols, "percentil", "prumerny %", "průměrný %", "prumer_jpz_bodu", "prumerny bod")
        if rid and rval:
            rs = pd.DataFrame({"redizo": res[rid].map(_digits), "actual_score_value": pd.to_numeric(res[rval], errors="coerce")})
            rs = rs.groupby(["redizo"], dropna=False)["actual_score_value"].mean().reset_index()
            merged = merged.merge(rs, on="redizo", how="left", suffixes=("", "_res"))
            if "actual_score_value_res" in merged.columns:
                merged["actual_score_value"] = merged["actual_score_value_res"]
                merged = merged.drop(columns=["actual_score_value_res"])
            merged["actual_score_field"] = rval
            n = _norm_text(rval)
            merged["actual_score_unit"] = "percentile" if "percentil" in n else "points"

    merged = _attach_school_type_fields(merged)
    return merged


def parse_mz_components(path: Path, year: int, variant: str, source_id: str) -> pd.DataFrame:
    def _parse_multi() -> pd.DataFrame:
        df = pd.read_excel(path, header=[0, 1])
        if not isinstance(df.columns, pd.MultiIndex):
            return pd.DataFrame()

        def lvl(col: object, idx: int) -> str:
            if isinstance(col, tuple) and len(col) > idx and col[idx] is not None:
                return str(col[idx])
            return str(col)

        cols = list(df.columns)

        def pick_obj(*candidates: str) -> object | None:
            for cand in candidates:
                n = _norm_text(cand)
                for col in cols:
                    if any(n == _norm_text(lvl(col, i)) or n in _norm_text(lvl(col, i)) for i in (0, 1)):
                        return col
            return None

        tier_col = pick_obj("třídění")
        redizo_col = pick_obj("redizo")
        name_col = pick_obj("název školy")
        addr_col = pick_obj("adresa školy")
        prog_col = pick_obj("smo16")
        kkov_col = pick_obj("kkov")
        prog_name_col = pick_obj("nazev oboru", "obor")
        prog_focus_col = pick_obj("zamereni", "zaměření", "specializace")
        year_col = pick_obj("rok")
        if redizo_col is None or name_col is None or tier_col is None:
            return pd.DataFrame()

        tier_values = df[tier_col].map(_norm_tier)
        row_mask = tier_values.eq("redizosmo16")
        if not row_mask.any():
            return pd.DataFrame()

        metric_cols: dict[tuple[str, str], object] = {}
        for col in cols:
            comp = _extract_component_from_text(lvl(col, 0))
            metric = _metric_from_text(lvl(col, 1))
            if comp in {"CJ", "M", "AJ"} and metric in {"sat", "mean_score", "pass_rate"}:
                metric_cols[(comp, metric)] = col

        recs: list[dict[str, Any]] = []
        for idx, row in df[row_mask].iterrows():
            redizo = _digits(row.get(redizo_col))
            if not redizo:
                continue
            name = _safe_text(row.get(name_col))
            address = _safe_text(row.get(addr_col)) if addr_col else None
            city, postcode = _city_postcode(address)
            yv = pd.to_numeric(pd.Series([row.get(year_col)]), errors="coerce").iloc[0] if year_col else year
            y = int(yv) if pd.notna(yv) else year
            for comp in ("CJ", "M", "AJ"):
                sat_col = metric_cols.get((comp, "sat"))
                score_col = metric_cols.get((comp, "mean_score"))
                pass_col = metric_cols.get((comp, "pass_rate"))
                if sat_col is None and score_col is None and pass_col is None:
                    continue
                sat = pd.to_numeric(pd.Series([row.get(sat_col)]), errors="coerce").iloc[0] if sat_col else pd.NA
                score = pd.to_numeric(pd.Series([row.get(score_col)]), errors="coerce").iloc[0] if score_col else pd.NA
                prate = pd.to_numeric(pd.Series([row.get(pass_col)]), errors="coerce").iloc[0] if pass_col else pd.NA
                if pd.notna(prate) and float(prate) > 1.0:
                    prate = float(prate) / 100.0
                recs.append(
                    {
                        "year": y,
                        "variant": variant,
                        "mz_tier_raw": row.get(tier_col),
                        "mz_tier_norm": "redizo_smo16",
                        "redizo": redizo,
                        "school_name_raw": name,
                        "address_raw": address,
                        "city": city,
                        "postcode": postcode,
                        "address_source_id": source_id if address is not None else pd.NA,
                        "city_source_id": source_id if city is not None else pd.NA,
                        "programme_group_raw": row.get(prog_col) if prog_col else pd.NA,
                        "programme_group_16_raw": row.get(prog_col) if prog_col else pd.NA,
                        "kkov_raw": row.get(kkov_col) if kkov_col else pd.NA,
                        "programme_name_raw": row.get(prog_name_col) if prog_name_col else pd.NA,
                        "programme_focus_raw": row.get(prog_focus_col) if prog_focus_col else pd.NA,
                        "grade": pd.NA,
                        "component": comp,
                        "candidates": sat,
                        "mean_score": score,
                        "pass_rate": prate,
                        "source_id": source_id,
                        "source_row_number": int(idx) + 3,
                    }
                )
        return pd.DataFrame(recs)

    def _parse_flat() -> pd.DataFrame:
        df = pd.read_excel(path, header=0)
        cols = [str(c) for c in df.columns]
        tier_col = _pick_col(cols, "trideni", "třídění")
        redizo_col = _pick_col(cols, "redizo")
        name_col = _pick_col(cols, "nazev")
        addr_col = _pick_col(cols, "adresa")
        prog_col = _pick_col(cols, "smo16")
        kkov_col = _pick_col(cols, "kkov", "kod oboru", "obor kod")
        prog_name_col = _pick_col(cols, "nazev oboru", "obor")
        prog_focus_col = _pick_col(cols, "zamereni", "zaměření", "specializace")
        year_col = _pick_col(cols, "rok")
        if redizo_col is None or name_col is None or tier_col is None:
            return pd.DataFrame()

        tier_values = df[tier_col].map(_norm_tier)
        row_mask = tier_values.eq("redizosmo16")
        if not row_mask.any():
            return pd.DataFrame()

        metric_cols: dict[tuple[str, str], str] = {}
        for col in cols:
            comp = _extract_component_from_text(col)
            metric = _metric_from_text(col)
            if comp in {"TOTAL", "CJ", "M", "AJ"} and metric in {"sat", "mean_score", "pass_rate"}:
                metric_cols[(comp, metric)] = col

        recs: list[dict[str, Any]] = []
        for idx, row in df[row_mask].iterrows():
            redizo = _digits(row.get(redizo_col))
            if not redizo:
                continue
            name = _safe_text(row.get(name_col))
            address = _safe_text(row.get(addr_col)) if addr_col else None
            city, postcode = _city_postcode(address)
            yv = pd.to_numeric(pd.Series([row.get(year_col)]), errors="coerce").iloc[0] if year_col else year
            y = int(yv) if pd.notna(yv) else year
            for comp in ("TOTAL", "CJ", "M", "AJ"):
                sat_col = metric_cols.get((comp, "sat"))
                score_col = metric_cols.get((comp, "mean_score"))
                pass_col = metric_cols.get((comp, "pass_rate"))
                if sat_col is None and score_col is None and pass_col is None:
                    continue
                sat = pd.to_numeric(pd.Series([row.get(sat_col)]), errors="coerce").iloc[0] if sat_col else pd.NA
                score = pd.to_numeric(pd.Series([row.get(score_col)]), errors="coerce").iloc[0] if score_col else pd.NA
                prate = pd.to_numeric(pd.Series([row.get(pass_col)]), errors="coerce").iloc[0] if pass_col else pd.NA
                if pd.notna(prate) and float(prate) > 1.0:
                    prate = float(prate) / 100.0
                recs.append(
                    {
                        "year": y,
                        "variant": variant,
                        "mz_tier_raw": row.get(tier_col),
                        "mz_tier_norm": "redizo_smo16",
                        "redizo": redizo,
                        "school_name_raw": name,
                        "address_raw": address,
                        "city": city,
                        "postcode": postcode,
                        "address_source_id": source_id if address is not None else pd.NA,
                        "city_source_id": source_id if city is not None else pd.NA,
                        "programme_group_raw": row.get(prog_col) if prog_col else pd.NA,
                        "programme_group_16_raw": row.get(prog_col) if prog_col else pd.NA,
                        "kkov_raw": row.get(kkov_col) if kkov_col else pd.NA,
                        "programme_name_raw": row.get(prog_name_col) if prog_name_col else pd.NA,
                        "programme_focus_raw": row.get(prog_focus_col) if prog_focus_col else pd.NA,
                        "grade": pd.NA,
                        "component": comp,
                        "candidates": sat,
                        "mean_score": score,
                        "pass_rate": prate,
                        "source_id": source_id,
                        "source_row_number": int(idx) + 2,
                    }
                )
        return pd.DataFrame(recs)

    out = _parse_multi()
    if out.empty:
        out = _parse_flat()
    out = _attach_school_type_fields(out)
    return out[out["school_type"].astype(str).ne("TOTAL_AGGREGATE")].reset_index(drop=True)


def _build_school_dimension(jpz_components: pd.DataFrame, jpz_modern: pd.DataFrame, mz_components: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for df in (jpz_components, jpz_modern, mz_components):
        if df is None or df.empty:
            continue
        cur = df[[c for c in ["redizo", "school_name_raw", "address_raw", "city", "postcode", "source_id", "school_type", "programme_taxonomy", "programme_identity", "classification_quality"] if c in df.columns]].copy()
        cur["source_id"] = cur.get("source_id", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["redizo"] = cur.get("redizo", pd.Series([pd.NA] * len(cur), index=cur.index)).map(_digits).astype("string")
        cur["school_name_raw"] = cur.get("school_name_raw", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["address_raw"] = cur.get("address_raw", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["city"] = cur.get("city", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["postcode"] = cur.get("postcode", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["school_type"] = cur.get("school_type", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["programme_taxonomy"] = cur.get("programme_taxonomy", cur.get("school_type", pd.Series([pd.NA] * len(cur), index=cur.index))).astype("string")
        cur["programme_identity"] = cur.get("programme_identity", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        cur["classification_quality"] = cur.get("classification_quality", pd.Series([pd.NA] * len(cur), index=cur.index)).astype("string")
        frames.append(cur)

    if not frames:
        return pd.DataFrame(
            columns=[
                "school_key",
                "identity_quality",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "canonical_source_id",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "provenance_count",
                "provenance_sources",
            ]
        )

    base = pd.concat(frames, ignore_index=True)
    base["norm_redizo"] = base["redizo"].map(_digits)
    base["norm_name"] = base["school_name_raw"].map(_norm_key_part)
    base["norm_addr"] = base["address_raw"].map(_norm_key_part)
    base["norm_city"] = base["city"].map(_norm_key_part)
    base["name_address_key"] = base.apply(
        lambda r: f"name_address:{r['norm_name']}|{r['norm_addr']}" if r["norm_name"] and r["norm_addr"] else pd.NA,
        axis=1,
    )
    base["name_city_key"] = base.apply(
        lambda r: f"name_city:{r['norm_name']}|{r['norm_city']}" if r["norm_name"] and r["norm_city"] else pd.NA,
        axis=1,
    )

    def _pick_best(g: pd.DataFrame, field: str) -> tuple[object, str | None]:
        if field == "school_type" and field in g.columns:
            values = g[[field, "source_id"]].copy()
            values["_text"] = values[field].map(_safe_text)
            values = values[values["_text"].notna()].copy()
            if not values.empty:
                values["_priority"] = values["_text"].map(
                    lambda x: _TAXONOMY_ORDER.index(str(x)) if str(x) in _TAXONOMY_ORDER else len(_TAXONOMY_ORDER)
                )
                values["_freq"] = values["_text"].map(values["_text"].value_counts(dropna=True).to_dict()).fillna(0).astype(int)
                values["_len"] = values["_text"].astype(str).str.len()
                values["_source_rank"] = values["source_id"].map(lambda s: {"jpz": 0, "mz": 1}.get(str(s).split("_")[0], 9))
                chosen = values.sort_values(["_priority", "_freq", "_len", "_source_rank", "source_id", "_text"], ascending=[True, False, False, True, True, True]).iloc[0]
                return chosen["_text"], str(chosen["source_id"])
        value, sid = _best_non_null(g, field)
        return value, sid

    na_redizo = (
        base[base["name_address_key"].notna() & base["norm_redizo"].notna()]
        .groupby("name_address_key", dropna=False)["norm_redizo"]
        .nunique()
    )
    nc_redizo = (
        base[base["name_city_key"].notna() & base["norm_redizo"].notna()]
        .groupby("name_city_key", dropna=False)["norm_redizo"]
        .nunique()
    )
    nc_addr = (
        base[base["name_city_key"].notna() & base["norm_addr"].astype(str).ne("")]
        .groupby("name_city_key", dropna=False)["norm_addr"]
        .nunique()
    )

    keys: list[str] = []
    quality: list[str] = []
    for _, row in base.iterrows():
        if row["norm_redizo"]:
            keys.append(f"redizo:{row['norm_redizo']}")
            quality.append("redizo")
            continue

        na_key = row.get("name_address_key")
        nc_key = row.get("name_city_key")
        if pd.notna(na_key) and int(na_redizo.get(na_key, 0)) <= 1:
            keys.append(str(na_key))
            quality.append("name_address")
            continue
        if pd.notna(nc_key) and int(nc_redizo.get(nc_key, 0)) <= 1 and int(nc_addr.get(nc_key, 0)) <= 1:
            keys.append(str(nc_key))
            quality.append("name_city")
            continue

        unresolved_basis = "|".join([row.get("norm_name", ""), row.get("norm_addr", ""), row.get("norm_city", "")])
        keys.append(f"unresolved:{hashlib.sha1(unresolved_basis.encode('utf-8')).hexdigest()[:16]}")
        quality.append("unresolved")

    base["school_key"] = keys
    base["identity_quality"] = quality

    base["_name_len"] = base["school_name_raw"].astype("string").str.len().fillna(0)
    base["_addr_len"] = base["address_raw"].astype("string").str.len().fillna(0)
    base["_city_len"] = base["city"].astype("string").str.len().fillna(0)
    base["_rank_quality"] = base["identity_quality"].map({"redizo": 0, "name_address": 1, "name_city": 2, "unresolved": 3}).fillna(9)

    dim_rows: list[dict[str, Any]] = []
    for school_key, g in base.groupby("school_key", dropna=False):
        provenance = sorted({str(x) for x in g["source_id"].dropna().astype(str).tolist() if x and x != "<NA>"})
        school_name_raw, school_name_source_id = _pick_best(g, "school_name_raw")
        address_raw, address_source_id = _pick_best(g, "address_raw")
        city, city_source_id = _pick_best(g, "city")
        postcode, postcode_source_id = _pick_best(g, "postcode")
        school_type, _ = _pick_best(g, "school_type")
        programme_taxonomy, _ = _pick_best(g, "programme_taxonomy")
        programme_identity, _ = _pick_best(g, "programme_identity")
        classification_quality, _ = _pick_best(g, "classification_quality")
        redizo_value = g["norm_redizo"].dropna().astype(str)
        redizo = redizo_value.value_counts().index[0] if not redizo_value.empty else pd.NA
        identity_quality = "redizo" if pd.notna(redizo) else g.sort_values(["_rank_quality", "_addr_len", "_city_len", "_name_len", "source_id"], ascending=[True, False, False, False, True]).iloc[0]["identity_quality"]
        canonical_source_id = address_source_id or school_name_source_id or city_source_id or postcode_source_id or (provenance[0] if provenance else pd.NA)
        dim_rows.append(
            {
                "school_key": school_key,
                "identity_quality": identity_quality,
                "redizo": redizo,
                "school_name_raw": school_name_raw,
                "address_raw": address_raw,
                "city": city,
                "postcode": postcode,
                "school_type": school_type,
                "programme_taxonomy": programme_taxonomy,
                "programme_identity": programme_identity,
                "classification_quality": classification_quality,
                "canonical_source_id": canonical_source_id,
                "school_name_source_id": school_name_source_id,
                "address_source_id": address_source_id,
                "city_source_id": city_source_id,
                "postcode_source_id": postcode_source_id,
                "provenance_count": len(provenance),
                "provenance_sources": "|".join(provenance),
            }
        )

    return pd.DataFrame(dim_rows).sort_values("school_key").reset_index(drop=True)


def _attach_school_identity(df: pd.DataFrame, school_dim: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["school_key"] = pd.Series(dtype="string")
        out["identity_quality"] = pd.Series(dtype="string")
        return out

    out = df.copy()
    out["_norm_redizo"] = out.get("redizo", pd.Series([pd.NA] * len(out), index=out.index)).map(_digits)
    out["_norm_name"] = out.get("school_name_raw", pd.Series([pd.NA] * len(out), index=out.index)).map(_norm_key_part)
    out["_norm_addr"] = out.get("address_raw", pd.Series([pd.NA] * len(out), index=out.index)).map(_norm_key_part)
    out["_norm_city"] = out.get("city", pd.Series([pd.NA] * len(out), index=out.index)).map(_norm_key_part)

    out["_name_address_key"] = out.apply(
        lambda r: f"name_address:{r['_norm_name']}|{r['_norm_addr']}" if r["_norm_name"] and r["_norm_addr"] else pd.NA,
        axis=1,
    )
    out["_name_city_key"] = out.apply(
        lambda r: f"name_city:{r['_norm_name']}|{r['_norm_city']}" if r["_norm_name"] and r["_norm_city"] else pd.NA,
        axis=1,
    )

    redizo_dim = school_dim[school_dim["identity_quality"].eq("redizo") & school_dim["redizo"].notna()][["redizo", "school_key", "identity_quality"]].drop_duplicates()
    na_dim = school_dim[school_dim["identity_quality"].eq("name_address")][["school_key", "identity_quality"]].drop_duplicates()
    na_dim["_name_address_key"] = na_dim["school_key"]
    nc_dim = school_dim[school_dim["identity_quality"].eq("name_city")][["school_key", "identity_quality"]].drop_duplicates()
    nc_dim["_name_city_key"] = nc_dim["school_key"]

    out = out.merge(redizo_dim, left_on="_norm_redizo", right_on="redizo", how="left", suffixes=("", "_d"))
    out = out.rename(columns={"school_key": "_school_key_redizo", "identity_quality": "_quality_redizo"})
    out = out.drop(columns=["redizo_d"] if "redizo_d" in out.columns else [], errors="ignore")

    out = out.merge(na_dim[["_name_address_key", "school_key", "identity_quality"]], on="_name_address_key", how="left", suffixes=("", "_na"))
    out = out.rename(columns={"school_key": "_school_key_na", "identity_quality": "_quality_na"})

    out = out.merge(nc_dim[["_name_city_key", "school_key", "identity_quality"]], on="_name_city_key", how="left", suffixes=("", "_nc"))
    out = out.rename(columns={"school_key": "_school_key_nc", "identity_quality": "_quality_nc"})

    out["school_key"] = out["_school_key_redizo"]
    out["identity_quality"] = out["_quality_redizo"]
    out.loc[out["school_key"].isna() & out["_school_key_na"].notna(), "school_key"] = out["_school_key_na"]
    out.loc[out["identity_quality"].isna() & out["_quality_na"].notna(), "identity_quality"] = out["_quality_na"]
    out.loc[out["school_key"].isna() & out["_school_key_nc"].notna(), "school_key"] = out["_school_key_nc"]
    out.loc[out["identity_quality"].isna() & out["_quality_nc"].notna(), "identity_quality"] = out["_quality_nc"]

    unresolved_mask = out["school_key"].isna()
    unresolved_basis = (
        out["_norm_name"].fillna("") + "|" + out["_norm_addr"].fillna("") + "|" + out["_norm_city"].fillna("")
    )
    out.loc[unresolved_mask, "school_key"] = unresolved_basis[unresolved_mask].map(
        lambda x: f"unresolved:{hashlib.sha1(str(x).encode('utf-8')).hexdigest()[:16]}"
    )
    out.loc[out["identity_quality"].isna(), "identity_quality"] = "unresolved"

    drop_cols = [
        "_norm_redizo",
        "_norm_name",
        "_norm_addr",
        "_norm_city",
        "_name_address_key",
        "_name_city_key",
        "_school_key_redizo",
        "_quality_redizo",
        "_school_key_na",
        "_quality_na",
        "_school_key_nc",
        "_quality_nc",
    ]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def _cohort_matched(jpz_components: pd.DataFrame, mz_components: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if jpz_components.empty or mz_components.empty:
        pd.DataFrame(
            columns=[
                "entry_year",
                "graduation_year",
                "cohort_lag_years",
                "identity_quality",
                "redizo",
                "school_name_raw_jpz",
                "address_raw_jpz",
                "city_jpz",
                "postcode_jpz",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "programme_group_raw_jpz",
                "programme_group_16_raw_jpz",
                "kkov_raw_jpz",
                "programme_name_raw_jpz",
                "programme_focus_raw_jpz",
                "component",
                "jpz_mean_percentile",
                "mz_mean_score_pct",
                "mz_school_mean_percentile",
                "mz_school_mean_percentile_method",
                "mz_school_mean_percentile_reference",
                "slope_mz_mean_score_pct_per_year",
                "slope_mz_school_mean_percentile_per_year",
                "mz_candidates",
                "variant",
                "synthetic_admitted_intake_selectivity_percentile",
                "synthetic_admitted_intake_selectivity_latent",
                "jpz_test_takers_cj_m_mean",
                "capacity_throughput_proxy_candidates",
                "synthetic_offer_fraction_central",
                "synthetic_uplift_central",
                "mz_participation_rate_vs_cj",
            ]
        ).to_csv(out_dir / "cohort_component_panel.csv", index=False)
        pd.DataFrame(
            columns=[
                "entry_year",
                "graduation_year",
                "school_key",
                "redizo",
                "school_name_raw_jpz",
                "address_raw_jpz",
                "city_jpz",
                "postcode_jpz",
                "school_type",
                "jpz_mean_percentile",
                "jpz_test_takers_cj_m_mean",
                "capacity_throughput_proxy_candidates",
                "synthetic_admitted_intake_selectivity_latent",
                "synthetic_admitted_intake_selectivity_percentile",
                "outcome_component",
                "outcome_mz_mean_score_pct",
                "outcome_mz_candidates",
                "outcome_mz_school_mean_percentile",
                "outcome_mz_school_mean_percentile_valid",
                "outcome_mz_school_mean_percentile_method",
                "outcome_mz_school_mean_percentile_reference",
                "outcome_mz_school_mean_percentile_n_schools",
                "scenario_selectivity_valid",
                "outcome_score_valid",
                "outcome_candidates_valid",
                "plot_eligible",
                "plot_exclusion_reason",
                "mz_participation_rate_vs_cj",
            ]
        ).to_csv(out_dir / "scenario_intake_vs_mz_outcomes.csv", index=False)
        pd.DataFrame(
            columns=[
                "scope",
                "entry_year",
                "graduation_year",
                "component",
                "n_schools",
                "weighted_slope_mz_pct_per_jpz_percentile",
                "pearson_correlation",
            ]
        ).to_csv(out_dir / "pooled_component_association.csv", index=False)
        dump_json(
            out_dir / "metadata.json",
            {
                "analysis_type": "school_level_association_not_causal",
                "join_rule": "school_key + school_type(SMO16 taxonomy category) + component + graduation_year=entry_year+programme_duration_years",
                "note": "No overlapping cohort-component rows available in local archive.",
                "scenario_intake_selectivity": {
                    "status": "empty",
                    "metric": "synthetic_admitted_intake_selectivity_percentile",
                    "label": "scenario intake selectivity percentile (CJ+M)",
                },
                "matching_diagnostics": {
                    "jpz_rows": 0,
                    "mz_rows": 0,
                    "joined_rows": 0,
                },
            },
        )
        return
    jpz = jpz_components[jpz_components["component"].isin(["CJ", "M"])].copy()
    mz = mz_components[(mz_components["component"].isin(["CJ", "M", "AJ"]))].copy()
    jpz = jpz[~jpz["identity_quality"].eq("unresolved")].copy()
    mz = mz[~mz["identity_quality"].eq("unresolved")].copy()

    jpz["programme_duration_years"] = pd.to_numeric(jpz.get("programme_duration_years"), errors="coerce")
    jpz["entrant_grade"] = pd.to_numeric(jpz.get("entrant_grade"), errors="coerce")
    jpz["grade"] = pd.to_numeric(jpz.get("grade"), errors="coerce")
    mz["programme_duration_years"] = pd.to_numeric(mz.get("programme_duration_years"), errors="coerce")
    mz["entrant_grade"] = pd.to_numeric(mz.get("entrant_grade"), errors="coerce")

    jpz_gym = jpz[
        jpz.get("school_type").isin(_GYM_MATCH_TYPES)
        & jpz["programme_duration_years"].isin([4, 6, 8])
    ].copy()
    mz_gym = mz[
        mz.get("school_type").isin(_GYM_MATCH_TYPES)
        & mz["programme_duration_years"].isin([4, 6, 8])
    ].copy()

    jpz_sos = jpz[
        jpz.get("school_type").isin(_SOS_MATCH_TYPES)
        & jpz["programme_duration_years"].eq(4)
        & jpz["entrant_grade"].eq(9)
        & jpz["grade"].eq(9)
        & jpz.get("source_id", pd.Series(["" for _ in range(len(jpz))], index=jpz.index)).astype(str).str.contains("_historic_vysledky")
    ].copy()
    mz_sos = mz[
        mz.get("school_type").isin(_SOS_MATCH_TYPES)
        & mz["programme_duration_years"].eq(4)
        & mz["entrant_grade"].eq(9)
    ].copy()

    jpz = pd.concat([jpz_gym, jpz_sos], ignore_index=True)
    mz = pd.concat([mz_gym, mz_sos], ignore_index=True)

    jpz["match_school_type"] = jpz.get("school_type").astype("string")
    mz["match_school_type"] = mz.get("school_type").astype("string")
    mz_pref = (
        mz.assign(_rank=mz["variant"].map({"jap": 0, "j": 1}).fillna(9))
        .sort_values(["year", "school_key", "match_school_type", "component", "_rank"])
        .drop_duplicates(["year", "school_key", "match_school_type", "component"], keep="first")
    )
    mz_pref = _attach_mz_school_mean_percentile(mz_pref)
    mz_participation_rate = _compute_mz_participation_rate_vs_cj(mz_pref)

    jpz["graduation_year"] = pd.to_numeric(jpz["entry_year"], errors="coerce") + pd.to_numeric(jpz["programme_duration_years"], errors="coerce")
    base = jpz.merge(
        mz_pref,
        left_on=["school_key", "match_school_type", "graduation_year", "component"],
        right_on=["school_key", "match_school_type", "year", "component"],
        how="inner",
        suffixes=("_jpz", "_mz"),
    )
    base = base.rename(columns={"mean_percentile": "jpz_mean_percentile", "mean_score": "mz_mean_score_pct", "candidates": "mz_candidates"})
    if "redizo_jpz" in base.columns and "redizo" not in base.columns:
        base["redizo"] = base["redizo_jpz"]
    if "redizo" not in base.columns:
        base["redizo"] = pd.NA
    if "redizo_mz" in base.columns:
        base["redizo"] = base["redizo"].where(base["redizo"].notna(), base["redizo_mz"])
    base["redizo"] = base["redizo"].where(base["redizo"].notna(), base["school_key"].map(_redizo_from_school_key))
    base["cohort_lag_years"] = _numeric_series(base, "programme_duration_years_jpz")
    base["outcome_score_valid"] = _numeric_series(base, "mz_mean_score_pct").notna()
    base["outcome_candidates_valid"] = _numeric_series(base, "mz_candidates").gt(0).fillna(False)
    scenario_selectivity = _build_scenario_selectivity_and_outcomes(jpz, mz_pref)
    scenario_selectivity_cols = [
        "jpz_cj_mean_percentile",
        "jpz_m_mean_percentile",
        "synthetic_input_cj_m_z_mean",
        "synthetic_offer_fraction_denominator_source",
        "synthetic_offer_fraction_low",
        "synthetic_offer_fraction_central",
        "synthetic_offer_fraction_high",
        "synthetic_uplift_low",
        "synthetic_uplift_central",
        "synthetic_uplift_high",
        "synthetic_admitted_intake_selectivity_latent_low",
        "synthetic_admitted_intake_selectivity_latent",
        "synthetic_admitted_intake_selectivity_latent_high",
        "synthetic_admitted_intake_selectivity_percentile",
        "synthetic_admitted_intake_selectivity_percentile_n_schools",
        "jpz_test_takers_cj_m_mean",
        "jpz_test_takers_cj_m_mean_method",
        "capacity_throughput_proxy_candidates",
        "capacity_throughput_proxy_component_source",
        "capacity_throughput_proxy_variant_source",
        "capacity_throughput_proxy_source_year",
        "capacity_throughput_proxy_source_id",
        "capacity_throughput_proxy_method",
    ]
    if not scenario_selectivity.empty:
        scenario_selectivity = scenario_selectivity[
            ["school_key", "school_type", "entry_year", "graduation_year"]
            + [c for c in scenario_selectivity_cols if c in scenario_selectivity.columns]
        ].drop_duplicates(["school_key", "school_type", "entry_year", "graduation_year"], keep="first")
        scenario_selectivity = scenario_selectivity.rename(columns={"school_type": "match_school_type"})
        base = base.merge(
            scenario_selectivity,
            on=["school_key", "match_school_type", "entry_year", "graduation_year"],
            how="left",
            suffixes=("", "_scenario_selectivity"),
        )
        for c in scenario_selectivity_cols:
            c2 = f"{c}_scenario_selectivity"
            if c2 in base.columns:
                base[c] = base[c].where(base[c].notna(), base[c2]) if c in base.columns else base[c2]
                base = base.drop(columns=[c2])

    fallback_keys = ["school_key", "match_school_type", "entry_year", "graduation_year"]
    fallback = (
        base[base["component"].isin(["CJ", "M"])].copy()
        .groupby(fallback_keys, dropna=False)
        .agg(
            jpz_cj_mean_percentile=("jpz_mean_percentile", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            jpz_m_mean_percentile=("jpz_mean_percentile", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            jpz_cj_registered=("registered", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            jpz_m_registered=("registered", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            jpz_cj_sat=("sat", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            jpz_m_sat=("sat", lambda s: pd.to_numeric(s, errors="coerce").iloc[0] if len(s) else pd.NA),
            capacity_throughput_proxy_candidates=("capacity_throughput_proxy_candidates", "max"),
        )
        .reset_index()
    )
    if not fallback.empty:
        fallback["jpz_test_takers_cj_m_mean"] = pd.NA
        cj_test_takers = fallback["jpz_cj_sat"].where(pd.to_numeric(fallback["jpz_cj_sat"], errors="coerce").gt(0), fallback["jpz_cj_registered"])
        m_test_takers = fallback["jpz_m_sat"].where(pd.to_numeric(fallback["jpz_m_sat"], errors="coerce").gt(0), fallback["jpz_m_registered"])
        tt_mask = pd.to_numeric(cj_test_takers, errors="coerce").notna() & pd.to_numeric(m_test_takers, errors="coerce").notna() & (pd.to_numeric(cj_test_takers, errors="coerce") > 0) & (pd.to_numeric(m_test_takers, errors="coerce") > 0)
        if tt_mask.any():
            fallback.loc[tt_mask, "jpz_test_takers_cj_m_mean"] = (pd.to_numeric(cj_test_takers[tt_mask], errors="coerce") + pd.to_numeric(m_test_takers[tt_mask], errors="coerce")) / 2.0
        cj_p = (pd.to_numeric(fallback["jpz_cj_mean_percentile"], errors="coerce") / 100.0).clip(_SCENARIO_P_MIN, 0.98)
        m_p = (pd.to_numeric(fallback["jpz_m_mean_percentile"], errors="coerce") / 100.0).clip(_SCENARIO_P_MIN, 0.98)
        valid_input = pd.to_numeric(fallback["jpz_cj_mean_percentile"], errors="coerce").notna() & pd.to_numeric(fallback["jpz_m_mean_percentile"], errors="coerce").notna()
        fallback["synthetic_input_cj_m_z_mean"] = pd.NA
        if valid_input.any():
            fallback.loc[valid_input, "synthetic_input_cj_m_z_mean"] = ((cj_p.map(NormalDist().inv_cdf) + m_p.map(NormalDist().inv_cdf)) / 2.0)[valid_input]
        ok = (
            pd.to_numeric(fallback["synthetic_input_cj_m_z_mean"], errors="coerce").notna()
            & pd.to_numeric(fallback["jpz_test_takers_cj_m_mean"], errors="coerce").notna()
            & pd.to_numeric(fallback["capacity_throughput_proxy_candidates"], errors="coerce").notna()
            & (pd.to_numeric(fallback["jpz_test_takers_cj_m_mean"], errors="coerce") > 0)
            & (pd.to_numeric(fallback["capacity_throughput_proxy_candidates"], errors="coerce") > 0)
        )
        fallback["synthetic_offer_fraction_central"] = pd.NA
        fallback["synthetic_uplift_central"] = pd.NA
        fallback["synthetic_admitted_intake_selectivity_latent"] = pd.NA
        if ok.any():
            a = pd.to_numeric(fallback.loc[ok, "jpz_test_takers_cj_m_mean"], errors="coerce")
            c = pd.to_numeric(fallback.loc[ok, "capacity_throughput_proxy_candidates"], errors="coerce")
            p = (c / (a * float(_SCENARIO_CONFIGS["central"]["yield"]))).clip(_SCENARIO_P_MIN, _SCENARIO_P_MAX)
            u = p.map(lambda pp: _scenario_uplift(p=float(pp), dispersion=float(_SCENARIO_CONFIGS["central"]["within_school_dispersion"])))
            z0 = pd.to_numeric(fallback.loc[ok, "synthetic_input_cj_m_z_mean"], errors="coerce")
            fallback.loc[ok, "synthetic_offer_fraction_central"] = p
            fallback.loc[ok, "synthetic_uplift_central"] = u
            fallback.loc[ok, "synthetic_admitted_intake_selectivity_latent"] = z0 + u
        pct, n = _equal_weight_percentile_rank(pd.to_numeric(fallback["synthetic_admitted_intake_selectivity_latent"], errors="coerce"))
        fallback["synthetic_admitted_intake_selectivity_percentile"] = pct
        fallback["synthetic_admitted_intake_selectivity_percentile_n_schools"] = n
        fallback["scenario_selectivity_valid"] = pd.to_numeric(fallback["synthetic_admitted_intake_selectivity_latent"], errors="coerce").notna() & pct.notna()
        base = base.merge(
            fallback[[
                "school_key",
                "match_school_type",
                "entry_year",
                "graduation_year",
                "jpz_test_takers_cj_m_mean",
                "capacity_throughput_proxy_candidates",
                "synthetic_input_cj_m_z_mean",
                "synthetic_offer_fraction_central",
                "synthetic_uplift_central",
                "synthetic_admitted_intake_selectivity_latent",
                "synthetic_admitted_intake_selectivity_percentile",
                "synthetic_admitted_intake_selectivity_percentile_n_schools",
                "scenario_selectivity_valid",
            ]],
            on=fallback_keys,
            how="left",
            suffixes=("", "_fallback"),
        )
        for c in [
            "jpz_test_takers_cj_m_mean",
            "capacity_throughput_proxy_candidates",
            "synthetic_input_cj_m_z_mean",
            "synthetic_offer_fraction_central",
            "synthetic_uplift_central",
            "synthetic_admitted_intake_selectivity_latent",
            "synthetic_admitted_intake_selectivity_percentile",
            "synthetic_admitted_intake_selectivity_percentile_n_schools",
            "scenario_selectivity_valid",
        ]:
            cf = f"{c}_fallback"
            if cf in base.columns:
                base[c] = base[c].where(base[c].notna(), base[cf]) if c in base.columns else base[cf]
                base = base.drop(columns=[cf])

    if base["jpz_test_takers_cj_m_mean"].isna().any():
        group_keys = ["school_key", "match_school_type", "entry_year", "graduation_year"]
        base["jpz_test_takers_cj_m_mean"] = pd.to_numeric(base["jpz_test_takers_cj_m_mean"], errors="coerce")
        base["synthetic_input_cj_m_z_mean"] = pd.to_numeric(base.get("synthetic_input_cj_m_z_mean"), errors="coerce")
        base["synthetic_offer_fraction_central"] = pd.to_numeric(base.get("synthetic_offer_fraction_central"), errors="coerce")
        base["synthetic_uplift_central"] = pd.to_numeric(base.get("synthetic_uplift_central"), errors="coerce")
        base["synthetic_admitted_intake_selectivity_latent"] = pd.to_numeric(base.get("synthetic_admitted_intake_selectivity_latent"), errors="coerce")
        for _, g in base.groupby(group_keys, dropna=False):
            if g["jpz_test_takers_cj_m_mean"].notna().any():
                continue

            def _count_for_component(component: str) -> float | None:
                rows = g[g["component"].eq(component)]
                if rows.empty:
                    return None
                row = rows.iloc[0]
                for field in ("sat", "registered"):
                    val = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
                    if pd.notna(val) and float(val) > 0:
                        return float(val)
                return None

            cj_count = _count_for_component("CJ")
            m_count = _count_for_component("M")
            if cj_count is None or m_count is None:
                continue

            tt = (cj_count + m_count) / 2.0
            cj_mean = pd.to_numeric(pd.Series([g.iloc[0].get("jpz_cj_mean_percentile")]), errors="coerce").iloc[0]
            m_mean = pd.to_numeric(pd.Series([g.iloc[0].get("jpz_m_mean_percentile")]), errors="coerce").iloc[0]
            if pd.isna(cj_mean) or pd.isna(m_mean):
                continue

            z0 = (NormalDist().inv_cdf(float(max(_SCENARIO_P_MIN, min(0.98, cj_mean / 100.0)))) + NormalDist().inv_cdf(float(max(_SCENARIO_P_MIN, min(0.98, m_mean / 100.0))))) / 2.0
            cproxy = pd.to_numeric(pd.Series([g.iloc[0].get("capacity_throughput_proxy_candidates")]), errors="coerce").iloc[0]
            if pd.isna(cproxy) or cproxy <= 0:
                continue
            p = float(min(_SCENARIO_P_MAX, max(_SCENARIO_P_MIN, cproxy / (tt * float(_SCENARIO_CONFIGS["central"]["yield"])))))
            uplift = _scenario_uplift(p=p, dispersion=float(_SCENARIO_CONFIGS["central"]["within_school_dispersion"]))
            latent = z0 + uplift
            idx = g.index
            base.loc[idx, "jpz_test_takers_cj_m_mean"] = tt
            base.loc[idx, "synthetic_input_cj_m_z_mean"] = z0
            base.loc[idx, "synthetic_offer_fraction_central"] = p
            base.loc[idx, "synthetic_uplift_central"] = uplift
            base.loc[idx, "synthetic_admitted_intake_selectivity_latent"] = latent
            base.loc[idx, "synthetic_admitted_intake_selectivity_percentile"] = pd.NA
            base.loc[idx, "scenario_selectivity_valid"] = True
    base["scenario_selectivity_valid"] = _numeric_series(base, "synthetic_admitted_intake_selectivity_percentile").notna()
    base["plot_eligible"] = base["scenario_selectivity_valid"] & base["outcome_score_valid"] & base["outcome_candidates_valid"]
    base["plot_exclusion_reason"] = pd.NA
    base.loc[~base["scenario_selectivity_valid"], "plot_exclusion_reason"] = "invalid_selectivity"
    base.loc[base["scenario_selectivity_valid"] & ~base["outcome_candidates_valid"], "plot_exclusion_reason"] = "invalid_outcome_candidates"
    base.loc[base["scenario_selectivity_valid"] & base["outcome_candidates_valid"] & ~base["outcome_score_valid"], "plot_exclusion_reason"] = "invalid_outcome_score"
    base["slope_mz_mean_score_pct_per_year"] = np.nan
    base["slope_mz_school_mean_percentile_per_year"] = np.nan
    if not base.empty:
        for (_, _, _), g in base.groupby(["school_key", "match_school_type", "component"], dropna=False):
            slope_score = _linear_slope(g["graduation_year"], g["mz_mean_score_pct"])
            slope_pct = _linear_slope(g["graduation_year"], g["mz_school_mean_percentile"])
            base.loc[g.index, "slope_mz_mean_score_pct_per_year"] = slope_score
            base.loc[g.index, "slope_mz_school_mean_percentile_per_year"] = slope_pct
    keep = [
        "school_key",
        "entry_year",
        "graduation_year",
        "cohort_lag_years",
        "identity_quality_jpz",
        "redizo",
        "school_name_raw_jpz",
        "address_raw_jpz",
        "city_jpz",
        "postcode_jpz",
        "school_type_jpz",
        "programme_taxonomy_jpz",
        "programme_identity",
        "classification_quality_jpz",
        "programme_group_raw_jpz",
        "programme_group_16_raw_jpz",
        "kkov_raw_jpz",
        "programme_name_raw_jpz",
        "programme_focus_raw_jpz",
        "component",
        "jpz_mean_percentile",
        "registered",
        "sat",
        "mz_mean_score_pct",
        "mz_school_mean_percentile",
        "mz_school_mean_percentile_method",
        "mz_school_mean_percentile_reference",
        "slope_mz_mean_score_pct_per_year",
        "slope_mz_school_mean_percentile_per_year",
        "mz_candidates",
        "variant",
    ]
    for col in keep:
        if col not in base.columns:
            base[col] = pd.NA
    scenario_outcomes = base[base["component"].isin(["CJ", "M", "AJ"])].copy()
    if scenario_outcomes.empty:
        for c in [
            "synthetic_admitted_intake_selectivity_percentile",
            "synthetic_admitted_intake_selectivity_latent",
            "jpz_test_takers_cj_m_mean",
            "capacity_throughput_proxy_candidates",
            "synthetic_offer_fraction_central",
            "synthetic_uplift_central",
        ]:
            base[c] = pd.NA

    scenario_cols = [
        "synthetic_admitted_intake_selectivity_percentile",
        "synthetic_admitted_intake_selectivity_latent",
        "jpz_test_takers_cj_m_mean",
        "jpz_test_takers_cj_m_mean_method",
        "capacity_throughput_proxy_candidates",
        "synthetic_offer_fraction_denominator_source",
        "synthetic_offer_fraction_central",
        "synthetic_uplift_central",
        "outcome_score_valid",
        "outcome_candidates_valid",
        "scenario_selectivity_valid",
        "plot_eligible",
        "plot_exclusion_reason",
        "mz_school_mean_percentile_valid",
    ]
    for col in scenario_cols:
        if col not in base.columns:
            base[col] = pd.NA
    for col in [
        "capacity_throughput_proxy_component_source",
        "capacity_throughput_proxy_variant_source",
        "capacity_throughput_proxy_source_year",
        "capacity_throughput_proxy_source_id",
        "capacity_throughput_proxy_method",
        "synthetic_input_cj_m_z_mean",
        "synthetic_offer_fraction_low",
        "synthetic_offer_fraction_high",
        "synthetic_uplift_low",
        "synthetic_uplift_high",
        "synthetic_admitted_intake_selectivity_latent_low",
        "synthetic_admitted_intake_selectivity_latent_high",
        "synthetic_admitted_intake_selectivity_percentile_n_schools",
    ]:
        if col not in base.columns:
            base[col] = pd.NA
    scenario_core = base.reindex(
        columns=[
            "school_key",
            "match_school_type",
            "scenario_school_type",
            "entry_year",
            "graduation_year",
            "redizo",
            "school_name_raw_jpz",
            "address_raw_jpz",
            "city_jpz",
            "postcode_jpz",
            "school_type_jpz",
            "programme_taxonomy_jpz",
            "programme_identity",
            "classification_quality_jpz",
            "jpz_cj_mean_percentile",
            "jpz_m_mean_percentile",
            "jpz_mean_percentile",
            "jpz_test_takers_cj_m_mean",
            "jpz_test_takers_cj_m_mean_method",
            "capacity_throughput_proxy_candidates",
            "capacity_throughput_proxy_component_source",
            "capacity_throughput_proxy_variant_source",
            "capacity_throughput_proxy_source_year",
            "capacity_throughput_proxy_source_id",
            "capacity_throughput_proxy_method",
            "synthetic_input_cj_m_z_mean",
            "synthetic_offer_fraction_denominator_source",
            "synthetic_offer_fraction_central",
            "synthetic_offer_fraction_low",
            "synthetic_offer_fraction_high",
            "synthetic_uplift_central",
            "synthetic_uplift_low",
            "synthetic_uplift_high",
            "synthetic_admitted_intake_selectivity_latent",
            "synthetic_admitted_intake_selectivity_latent_low",
            "synthetic_admitted_intake_selectivity_latent_high",
            "synthetic_admitted_intake_selectivity_percentile",
            "synthetic_admitted_intake_selectivity_percentile_n_schools",
            "scenario_selectivity_valid",
            "outcome_score_valid",
            "outcome_candidates_valid",
            "plot_eligible",
            "plot_exclusion_reason",
        ]
    ).drop_duplicates(["school_key", "match_school_type", "entry_year", "graduation_year"]).copy()
    scenario_core["scenario_school_type"] = scenario_core["match_school_type"]

    panel = base[keep + scenario_cols].rename(
        columns={
            "identity_quality_jpz": "identity_quality",
            "city_jpz": "city",
            "postcode_jpz": "postcode",
            "school_type_jpz": "school_type",
            "programme_taxonomy_jpz": "programme_taxonomy",
            "classification_quality_jpz": "classification_quality",
            "registered": "jpz_registered",
            "sat": "jpz_sat",
        }
    )
    panel = panel.merge(
        mz_participation_rate.rename(columns={"year": "graduation_year", "match_school_type": "school_type"}),
        on=["school_key", "school_type", "graduation_year", "component"],
        how="left",
    )
    panel = panel.drop(columns=["school_key"])

    panel.to_csv(out_dir / "cohort_component_panel.csv", index=False)
    outcomes = mz_pref[mz_pref["component"].isin(["CJ", "M", "AJ"])].copy()
    outcomes = outcomes.rename(
        columns={
            "year": "graduation_year",
            "component": "outcome_component",
            "mean_score": "outcome_mz_mean_score_pct",
            "candidates": "outcome_mz_candidates",
            "mz_school_mean_percentile": "outcome_mz_school_mean_percentile",
            "mz_school_mean_percentile_valid": "outcome_mz_school_mean_percentile_valid",
            "variant": "outcome_variant",
            "source_id": "outcome_source_id",
            "match_school_type": "outcome_school_type",
        }
    )
    for col in [
        "outcome_component",
        "outcome_mz_mean_score_pct",
        "outcome_mz_candidates",
        "outcome_mz_school_mean_percentile",
        "outcome_mz_school_mean_percentile_valid",
        "outcome_variant",
        "outcome_source_id",
        "outcome_school_type",
        "graduation_year",
    ]:
        if col not in outcomes.columns:
            outcomes[col] = pd.NA
    scenario_outcomes = scenario_core.merge(
        outcomes,
        left_on=["school_key", "graduation_year", "scenario_school_type"],
        right_on=["school_key", "graduation_year", "outcome_school_type"],
        how="inner",
        suffixes=("_scenario", "_outcome"),
    )
    scenario_outcomes["redizo"] = (
        scenario_outcomes.get("redizo_scenario", pd.Series([pd.NA] * len(scenario_outcomes), index=scenario_outcomes.index))
        .where(
            scenario_outcomes.get("redizo_scenario", pd.Series([pd.NA] * len(scenario_outcomes), index=scenario_outcomes.index)).notna(),
            scenario_outcomes.get("redizo_outcome", pd.Series([pd.NA] * len(scenario_outcomes), index=scenario_outcomes.index)),
        )
    )
    scenario_outcomes["redizo"] = scenario_outcomes["redizo"].where(
        scenario_outcomes["redizo"].notna(),
        scenario_outcomes["school_key"].map(_redizo_from_school_key),
    )
    component_summary = (
        base[base["component"].isin(["CJ", "M"])][["school_key", "match_school_type", "entry_year", "graduation_year", "component", "jpz_mean_percentile"]]
        .dropna(subset=["jpz_mean_percentile"])
        .drop_duplicates(["school_key", "match_school_type", "entry_year", "graduation_year", "component"], keep="first")
        .pivot_table(
            index=["school_key", "match_school_type", "entry_year", "graduation_year"],
            columns="component",
            values="jpz_mean_percentile",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"CJ": "jpz_cj_mean_percentile", "M": "jpz_m_mean_percentile"})
    )
    scenario_outcomes = scenario_outcomes.merge(
        component_summary,
        left_on=["school_key", "scenario_school_type", "entry_year", "graduation_year"],
        right_on=["school_key", "match_school_type", "entry_year", "graduation_year"],
        how="left",
        suffixes=("", "_component"),
    )
    if "match_school_type" in scenario_outcomes.columns:
        scenario_outcomes = scenario_outcomes.drop(columns=["match_school_type"])
    scenario_outcomes["school_type"] = scenario_outcomes["scenario_school_type"]
    scenario_outcome_cols = [
        "school_key",
        "school_type",
        "scenario_school_type",
        "outcome_school_type",
        "entry_year",
        "graduation_year",
        "outcome_component",
        "redizo",
        "school_name_raw_jpz",
        "address_raw_jpz",
        "city_jpz",
        "postcode_jpz",
        "programme_taxonomy_jpz",
        "programme_identity",
        "classification_quality_jpz",
        "jpz_mean_percentile",
        "jpz_cj_mean_percentile",
        "jpz_m_mean_percentile",
        "jpz_test_takers_cj_m_mean",
        "jpz_test_takers_cj_m_mean_method",
        "capacity_throughput_proxy_candidates",
        "capacity_throughput_proxy_component_source",
        "capacity_throughput_proxy_variant_source",
        "capacity_throughput_proxy_source_year",
        "capacity_throughput_proxy_source_id",
        "capacity_throughput_proxy_method",
        "synthetic_input_cj_m_z_mean",
        "synthetic_offer_fraction_denominator_source",
        "synthetic_offer_fraction_central",
        "synthetic_offer_fraction_low",
        "synthetic_offer_fraction_high",
        "synthetic_uplift_central",
        "synthetic_uplift_low",
        "synthetic_uplift_high",
        "synthetic_admitted_intake_selectivity_latent",
        "synthetic_admitted_intake_selectivity_latent_low",
        "synthetic_admitted_intake_selectivity_latent_high",
        "synthetic_admitted_intake_selectivity_percentile",
        "synthetic_admitted_intake_selectivity_percentile_n_schools",
        "scenario_selectivity_valid",
        "outcome_mz_mean_score_pct",
        "outcome_mz_school_mean_percentile",
        "outcome_mz_school_mean_percentile_valid",
        "outcome_mz_candidates",
        "outcome_score_valid",
        "outcome_candidates_valid",
        "plot_eligible",
        "plot_exclusion_reason",
        "outcome_variant",
        "outcome_source_id",
        "mz_participation_rate_vs_cj",
    ]
    scenario_outcomes = scenario_outcomes.merge(
        mz_participation_rate.rename(
            columns={"year": "graduation_year", "match_school_type": "outcome_school_type", "component": "outcome_component"}
        ),
        on=["school_key", "outcome_school_type", "graduation_year", "outcome_component"],
        how="left",
    )
    scenario_outcome_cols = list(dict.fromkeys(scenario_outcome_cols))
    for col in scenario_outcome_cols:
        if col not in scenario_outcomes.columns:
            scenario_outcomes[col] = pd.NA
    scenario_outcomes = scenario_outcomes[scenario_outcome_cols].copy()
    scenario_outcomes.to_csv(out_dir / "scenario_intake_vs_mz_outcomes.csv", index=False)

    stats: list[dict[str, Any]] = []
    for (entry, grad, comp), g in base.groupby(["entry_year", "graduation_year", "component"], dropna=False):
        x = pd.to_numeric(g["jpz_mean_percentile"], errors="coerce")
        y = pd.to_numeric(g["mz_mean_score_pct"], errors="coerce")
        w = pd.to_numeric(g["mz_candidates"], errors="coerce").fillna(1.0)
        mask = x.notna() & y.notna()
        slope = None
        corr = None
        mz_school_pct_slope = None
        if int(mask.sum()) >= 2:
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            wv = w[mask].to_numpy(dtype=float)
            slope = float(np.polyfit(xv, yv, 1, w=wv)[0])
            corr = float(pd.Series(xv).corr(pd.Series(yv)))
            mz_pct = pd.to_numeric(g["mz_school_mean_percentile"], errors="coerce")
            if mz_pct.notna().sum() >= 2:
                mz_school_pct_slope = float(np.polyfit(xv, mz_pct[mask].to_numpy(dtype=float), 1, w=wv)[0])
        stats.append(
            {
                "scope": "cohort_specific",
                "entry_year": entry,
                "graduation_year": grad,
                "component": comp,
                "n_schools": int(mask.sum()),
                "weighted_slope_mz_pct_per_jpz_percentile": slope,
                "weighted_slope_mz_school_mean_percentile_per_jpz_percentile": mz_school_pct_slope,
                "pearson_correlation": corr,
            }
        )

    for comp, g in base.groupby(["component"], dropna=False):
        x = pd.to_numeric(g["jpz_mean_percentile"], errors="coerce")
        y = pd.to_numeric(g["mz_mean_score_pct"], errors="coerce")
        w = pd.to_numeric(g["mz_candidates"], errors="coerce").fillna(1.0)
        mask = x.notna() & y.notna()
        slope = None
        corr = None
        mz_school_pct_slope = None
        if int(mask.sum()) >= 2:
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            wv = w[mask].to_numpy(dtype=float)
            slope = float(np.polyfit(xv, yv, 1, w=wv)[0])
            corr = float(pd.Series(xv).corr(pd.Series(yv)))
            mz_pct = pd.to_numeric(g["mz_school_mean_percentile"], errors="coerce") if "mz_school_mean_percentile" in g.columns else pd.Series(dtype=float)
            if mz_pct.notna().sum() >= 2:
                mz_school_pct_slope = float(np.polyfit(xv, mz_pct[mask].to_numpy(dtype=float), 1, w=wv)[0])
        stats.append(
            {
                "scope": "pooled_with_cohort_flags",
                "entry_year": "all",
                "graduation_year": "all",
                "component": comp,
                "n_schools": int(mask.sum()),
                "weighted_slope_mz_pct_per_jpz_percentile": slope,
                "weighted_slope_mz_school_mean_percentile_per_jpz_percentile": mz_school_pct_slope,
                "pearson_correlation": corr,
            }
        )

    for (ptype, comp), g in base.groupby(["programme_identity", "component"], dropna=False):
        x = pd.to_numeric(g["jpz_mean_percentile"], errors="coerce")
        y = pd.to_numeric(g["mz_mean_score_pct"], errors="coerce")
        w = pd.to_numeric(g["mz_candidates"], errors="coerce").fillna(1.0)
        mask = x.notna() & y.notna()
        slope = None
        corr = None
        mz_school_pct_slope = None
        if int(mask.sum()) >= 2:
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            wv = w[mask].to_numpy(dtype=float)
            slope = float(np.polyfit(xv, yv, 1, w=wv)[0])
            corr = float(pd.Series(xv).corr(pd.Series(yv)))
            mz_pct = pd.to_numeric(g["mz_school_mean_percentile"], errors="coerce") if "mz_school_mean_percentile" in g.columns else pd.Series(dtype=float)
            if mz_pct.notna().sum() >= 2:
                mz_school_pct_slope = float(np.polyfit(xv, mz_pct[mask].to_numpy(dtype=float), 1, w=wv)[0])
        stats.append(
            {
                "scope": f"pooled_programme_identity:{ptype}",
                "entry_year": "all",
                "graduation_year": "all",
                "component": comp,
                "programme_identity": ptype,
                "n_schools": int(mask.sum()),
                "weighted_slope_mz_pct_per_jpz_percentile": slope,
                "weighted_slope_mz_school_mean_percentile_per_jpz_percentile": mz_school_pct_slope,
                "pearson_correlation": corr,
            }
        )

    for (stype, comp), g in base.groupby(["match_school_type", "component"], dropna=False):
        x = pd.to_numeric(g["jpz_mean_percentile"], errors="coerce")
        y = pd.to_numeric(g["mz_mean_score_pct"], errors="coerce")
        w = pd.to_numeric(g["mz_candidates"], errors="coerce").fillna(1.0)
        mask = x.notna() & y.notna()
        slope = None
        corr = None
        mz_school_pct_slope = None
        if int(mask.sum()) >= 2:
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            wv = w[mask].to_numpy(dtype=float)
            slope = float(np.polyfit(xv, yv, 1, w=wv)[0])
            corr = float(pd.Series(xv).corr(pd.Series(yv)))
            mz_pct = pd.to_numeric(g["mz_school_mean_percentile"], errors="coerce") if "mz_school_mean_percentile" in g.columns else pd.Series(dtype=float)
            if mz_pct.notna().sum() >= 2:
                mz_school_pct_slope = float(np.polyfit(xv, mz_pct[mask].to_numpy(dtype=float), 1, w=wv)[0])
        stats.append(
            {
                "scope": f"pooled_school_type:{stype}",
                "entry_year": "all",
                "graduation_year": "all",
                "component": comp,
                "school_type": stype,
                "n_schools": int(mask.sum()),
                "weighted_slope_mz_pct_per_jpz_percentile": slope,
                "weighted_slope_mz_school_mean_percentile_per_jpz_percentile": mz_school_pct_slope,
                "pearson_correlation": corr,
            }
        )

    pd.DataFrame(stats).to_csv(out_dir / "pooled_component_association.csv", index=False)

    present_jpz_years = sorted({int(y) for y in pd.to_numeric(jpz["entry_year"], errors="coerce").dropna().astype(int).tolist()})
    present_mz_years = sorted({int(y) for y in pd.to_numeric(mz_pref["year"], errors="coerce").dropna().astype(int).tolist()})
    expected_by_year = sorted(
        [
            {
                "entry_year": int(r["entry_year"]),
                "graduation_year": int(r["graduation_year"]),
                "programme_identity": str(r["programme_identity"]),
                "school_type": str(r.get("match_school_type")),
                "cohort_lag_years": int(r["cohort_lag_years"]),
            }
            for _, r in base[["entry_year", "graduation_year", "programme_identity", "match_school_type", "cohort_lag_years"]].dropna().drop_duplicates().iterrows()
        ],
        key=lambda x: (x["entry_year"], x["graduation_year"], x.get("school_type") or "", x["programme_identity"]),
    )

    match_diag = {
        "jpz_rows": int(len(jpz)),
        "mz_rows": int(len(mz_pref)),
        "joined_rows": int(len(panel)),
        "jpz_schools": int(jpz["school_key"].nunique()),
        "mz_schools": int(mz_pref["school_key"].nunique()),
        "joined_schools": int(base["school_key"].nunique()) if not base.empty else 0,
        "jpz_components": {k: int(v) for k, v in jpz["component"].value_counts(dropna=False).to_dict().items()},
        "mz_components": {k: int(v) for k, v in mz_pref["component"].value_counts(dropna=False).to_dict().items()},
        "joined_components": {k: int(v) for k, v in panel["component"].value_counts(dropna=False).to_dict().items()} if not panel.empty else {},
        "jpz_years_present": present_jpz_years,
        "mz_years_present": present_mz_years,
    }

    meta = {
        "analysis_type": "school_level_association_not_causal",
        "cohort_matching_scope": "restricted_to_GY4_GY6_GY8_and_defensible_4year_SOS_only",
        "cohort_matching_limitation": "Cohort and expected-vs-observed models use GY4/GY6/GY8 and defensible SOS 4-year (entrant grade 9) rows only; SOS matching is school × SMO16-category and not KKOV/programme-level.",
        "cohorts_expected": expected_by_year,
        "units": {
            "jpz_mean_percentile": "percentile",
            "mz_mean_score_pct": "percentage_points",
        },
        "join_rule": "school_key + school_type(SMO16 taxonomy category) + component + graduation_year=entry_year+programme_duration_years",
        "aj_note": "AJ is descriptive only in maturita outputs; no JPZ baseline exists for AJ.",
        "scenario_intake_selectivity": {
            "metric": "synthetic_admitted_intake_selectivity_percentile",
            "label": "scenario intake selectivity percentile (Cproxy/CJ+M)",
            "family": "scenario_proxy_bounds_not_identified_preference_network",
            "non_identifiability_notice": "School-to-school two-choice preference/assignment cannot be reconstructed from school aggregates. Scenario assumptions adjust effective offers/yield and do not recover preference networks, accepted counts, acceptance rates, or causal effects.",
            "capacity_proxy_notice": "Historic seats are unavailable. Capacity proxy equals observed MZ cohort throughput (max candidates across CJ/M/AJ for same school_type and graduation_year). If proxy missing, synthetic metric is null.",
            "constants": {
                "offer_fraction_bounds": {"min": _SCENARIO_P_MIN, "max": _SCENARIO_P_MAX},
                "scenarios": {
                    name: {
                        "yield": cfg["yield"],
                        "within_school_dispersion": cfg["within_school_dispersion"],
                    }
                    for name, cfg in _SCENARIO_CONFIGS.items()
                },
                "uplift_formula": "uplift = within_school_dispersion * phi(Phi^-1(1-p)) / p",
                "p_formula": "p = clip(capacity_proxy / (jpz_test_takers_cj_m_mean * yield), 0.02, 1.0)",
                "input_mapping": "Cproxy/JPZ mean percentile clipped to [0.02,0.98], mapped with inverse normal CDF, equal-weight z-average",
            },
            "output_scope": "within entry_year × school_type percentile rank only; not an absolute scale across school types",
            "aj_outcome_note": "AJ is outcome-only descriptive MZ score paired with the same CJ+M synthetic input when AJ rows exist.",
            "output_file": "scenario_intake_vs_mz_outcomes.csv",
            "invalid_row_counts": {
                str(k) if k is not pd.NA else "<NA>": int(v)
                for k, v in base["plot_exclusion_reason"].fillna("<NA>").value_counts(dropna=False).to_dict().items()
            },
        },
        "invalid_metrics_not_used": ["CELKEM-derived JPZ entry metric"],
        "sos_match_types": sorted(_SOS_MATCH_TYPES),
        "sos_match_rule": {
            "required_programme_duration_years": 4,
            "required_entrant_grade": 9,
            "required_jpz_historic_grade": 9,
            "required_jpz_source_family": "historic_vysledky",
            "matching_granularity": "school × SMO16-category",
            "not_programme_level": True,
        },
        "matching_diagnostics": match_diag,
    }
    dump_json(out_dir / "metadata.json", meta)


def _expected_vs_observed_association(cohort_panel: pd.DataFrame, out_dir: Path, min_candidates: int = 10) -> None:
    ensure_dir(out_dir)
    out_csv = out_dir / "expected_vs_observed_association.csv"
    out_meta = out_dir / "expected_vs_observed_metadata.json"

    required_cols = {
        "entry_year",
        "graduation_year",
        "identity_quality",
        "redizo",
        "school_name_raw_jpz",
        "address_raw_jpz",
        "city",
        "postcode",
        "school_type",
        "programme_taxonomy",
        "programme_identity",
        "classification_quality",
        "programme_group_raw_jpz",
        "programme_group_16_raw_jpz",
        "kkov_raw_jpz",
        "programme_name_raw_jpz",
        "programme_focus_raw_jpz",
        "component",
        "jpz_mean_percentile",
        "mz_mean_score_pct",
        "mz_school_mean_percentile",
        "mz_candidates",
    }
    if cohort_panel.empty or not required_cols.issubset(set(cohort_panel.columns)):
        pd.DataFrame(
            columns=[
                "component",
                "entry_year",
                "graduation_year",
                "identity_quality",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "programme_group_raw",
                "programme_group_16_raw",
                "kkov_raw",
                "programme_name_raw",
                "programme_focus_raw",
                "jpz_mean_percentile",
                "mz_mean_score_pct_observed",
                "mz_mean_score_pct_expected",
                "mz_school_mean_percentile",
                "residual_pp",
                "mz_candidates",
                "quality_flag",
                "formula",
                "model_r_weighted",
                "beta_jpz_pp_per_percentile",
                "intercept_pp",
            ]
        ).to_csv(out_csv, index=False)
        dump_json(
            out_meta,
            {
                "analysis_type": "descriptive_school_level_association",
                "note": "No eligible matched CJ/M cohort rows available.",
                "minimum_candidates_threshold": int(min_candidates),
                "warnings": [
                    "Descriptive school-level association only; no individual linkage or causal inference.",
                    "Historic admissions/accepted and unique applicant adjustment are unavailable in source files.",
                ],
            },
        )
        return

    panel = cohort_panel.copy()
    panel = panel[panel["component"].isin(["CJ", "M"])].copy()
    panel["jpz_mean_percentile"] = pd.to_numeric(panel["jpz_mean_percentile"], errors="coerce")
    panel["mz_mean_score_pct"] = pd.to_numeric(panel["mz_mean_score_pct"], errors="coerce")
    panel["mz_candidates"] = pd.to_numeric(panel["mz_candidates"], errors="coerce")
    panel["entry_year"] = pd.to_numeric(panel["entry_year"], errors="coerce").astype("Int64")
    panel["graduation_year"] = pd.to_numeric(panel["graduation_year"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["jpz_mean_percentile", "mz_mean_score_pct", "mz_candidates", "entry_year", "graduation_year"])

    results: list[dict[str, Any]] = []
    model_meta: dict[str, Any] = {"components": {}}

    for comp in ("CJ", "M"):
        cdf = panel[panel["component"] == comp].copy()
        if cdf.empty:
            model_meta["components"][comp] = {"rows": 0, "status": "empty"}
            continue

        cohorts = sorted({int(x) for x in cdf["entry_year"].dropna().astype(int).tolist()})
        if not cohorts:
            model_meta["components"][comp] = {"rows": 0, "status": "empty_cohort"}
            continue

        cdf = cdf.sort_values(["entry_year", "redizo", "school_name_raw_jpz"]).reset_index(drop=True)
        w = cdf["mz_candidates"].astype(float).to_numpy()
        x_jpz = cdf["jpz_mean_percentile"].astype(float).to_numpy()
        y_mz = cdf["mz_mean_score_pct"].astype(float).to_numpy()

        cohort_to_col = {coh: i for i, coh in enumerate(cohorts[1:])}
        X = np.zeros((len(cdf), 2 + len(cohort_to_col)), dtype=float)
        X[:, 0] = 1.0
        X[:, 1] = x_jpz
        for i, coh in enumerate(cdf["entry_year"].astype(int).tolist()):
            j = cohort_to_col.get(coh)
            if j is not None:
                X[i, 2 + j] = 1.0

        sw = np.sqrt(np.maximum(w, 0.0))
        Xw = X * sw[:, None]
        yw = y_mz * sw
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        y_hat = X @ beta
        resid = y_mz - y_hat
        mz_school_pct_slope = None
        if "mz_school_mean_percentile" in cdf.columns:
            mz_pct = pd.to_numeric(cdf["mz_school_mean_percentile"], errors="coerce").to_numpy(dtype=float)
            pct_mask = np.isfinite(x_jpz) & np.isfinite(mz_pct)
            if int(pct_mask.sum()) >= 2:
                mz_school_pct_slope = float(np.polyfit(x_jpz[pct_mask], mz_pct[pct_mask], 1, w=np.maximum(w[pct_mask], 0.0))[0])

        y_mean_w = float(np.average(y_mz, weights=np.maximum(w, 1e-9)))
        sse = float(np.sum(np.maximum(w, 0.0) * ((y_mz - y_hat) ** 2)))
        sst = float(np.sum(np.maximum(w, 0.0) * ((y_mz - y_mean_w) ** 2)))
        r2 = None if sst <= 0 else float(1.0 - (sse / sst))
        r_weighted = None
        if sst > 0 and np.sum(np.maximum(w, 0.0)) > 0:
            xw = np.average(x_jpz, weights=np.maximum(w, 1e-9))
            yw_mean = np.average(y_mz, weights=np.maximum(w, 1e-9))
            cov = np.average((x_jpz - xw) * (y_mz - yw_mean), weights=np.maximum(w, 1e-9))
            varx = np.average((x_jpz - xw) ** 2, weights=np.maximum(w, 1e-9))
            vary = np.average((y_mz - yw_mean) ** 2, weights=np.maximum(w, 1e-9))
            if varx > 0 and vary > 0:
                r_weighted = float(cov / np.sqrt(varx * vary))

        jpz_sd = float(np.std(x_jpz, ddof=1)) if len(x_jpz) > 1 else 0.0
        mz_sd = float(np.std(y_mz, ddof=1)) if len(y_mz) > 1 else 0.0

        low_spread = comp == "M" and jpz_sd < 5.0
        coef_intercept = float(beta[0])
        coef_jpz = float(beta[1])

        for i, row in cdf.reset_index(drop=True).iterrows():
            entry = int(row["entry_year"])
            cohort_adj = 0.0 if entry == cohorts[0] else float(beta[2 + cohort_to_col.get(entry, 0)])
            quality = "ok"
            if float(row["mz_candidates"]) < float(min_candidates):
                quality = "low_candidates"
            if low_spread:
                quality = "unstable_low_jpz_spread" if quality == "ok" else f"{quality}|unstable_low_jpz_spread"
            results.append(
                {
                    "component": comp,
                    "entry_year": int(row["entry_year"]),
                    "graduation_year": int(row["graduation_year"]),
                    "identity_quality": row["identity_quality"],
                    "redizo": row["redizo"],
                    "school_name_raw": row.get("school_name_raw_jpz"),
                    "address_raw": row.get("address_raw_jpz"),
                    "city": row.get("city"),
                    "postcode": row.get("postcode"),
                    "school_type": row.get("school_type"),
                    "programme_taxonomy": row.get("programme_taxonomy"),
                    "programme_identity": row.get("programme_identity"),
                    "classification_quality": row.get("classification_quality"),
                    "programme_group_raw": row.get("programme_group_raw_jpz"),
                    "programme_group_16_raw": row.get("programme_group_16_raw_jpz"),
                    "kkov_raw": row.get("kkov_raw_jpz"),
                    "programme_name_raw": row.get("programme_name_raw_jpz"),
                    "programme_focus_raw": row.get("programme_focus_raw_jpz"),
                    "jpz_mean_percentile": float(row["jpz_mean_percentile"]),
                    "mz_mean_score_pct_observed": float(row["mz_mean_score_pct"]),
                    "mz_mean_score_pct_expected": float(y_hat[i]),
                    "mz_school_mean_percentile": float(row["mz_school_mean_percentile"]) if pd.notna(row.get("mz_school_mean_percentile")) else pd.NA,
                    "residual_pp": float(resid[i]),
                    "mz_candidates": float(row["mz_candidates"]),
                    "quality_flag": quality,
                    "formula": "MZ_score_pct ~ JPZ_published_mean_percentile + entry_year_fixed_effect",
                    "component_sd_jpz": jpz_sd,
                    "component_sd_mz": mz_sd,
                    "model_r2_weighted": r2,
                    "model_r_weighted": r_weighted,
                    "beta_jpz_pp_per_percentile": coef_jpz,
                    "intercept_pp": coef_intercept,
                    "cohort_effect_pp": cohort_adj,
                    "cohort_base_entry_year": cohorts[0],
                    "min_candidates_threshold": int(min_candidates),
                }
            )

        model_meta["components"][comp] = {
            "rows": int(len(cdf)),
            "cohorts": cohorts,
            "weighted_r2": r2,
            "weighted_r": r_weighted,
            "beta_jpz_pp_per_percentile": coef_jpz,
            "mz_school_mean_percentile_slope_per_jpz_percentile": mz_school_pct_slope,
            "intercept_pp": coef_intercept,
            "jpz_sd": jpz_sd,
            "mz_sd": mz_sd,
            "low_spread_warning": bool(low_spread),
        }

    out_df = pd.DataFrame(results)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "component",
                "entry_year",
                "graduation_year",
                "identity_quality",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "programme_group_raw",
                "programme_group_16_raw",
                "kkov_raw",
                "programme_name_raw",
                "programme_focus_raw",
                "jpz_mean_percentile",
                "mz_mean_score_pct_observed",
                "mz_mean_score_pct_expected",
                "mz_school_mean_percentile",
                "residual_pp",
                "mz_candidates",
                "quality_flag",
                "formula",
                "component_sd_jpz",
                "component_sd_mz",
                "model_r2_weighted",
                "model_r_weighted",
                "beta_jpz_pp_per_percentile",
                "intercept_pp",
                "cohort_effect_pp",
                "cohort_base_entry_year",
                "min_candidates_threshold",
            ]
        )
    out_df.to_csv(out_csv, index=False)

    dump_json(
        out_meta,
        {
            "analysis_type": "descriptive_school_level_association",
            "formula": "MZ score (%) ~ JPZ published mean percentile + entry-year cohort fixed effect",
            "weights": "mz_candidates",
            "units": {
                "jpz_mean_percentile": "percentile",
                "mz_mean_score_pct": "percentage_points",
                "residual_pp": "percentage_points",
            },
            "minimum_candidates_threshold": int(min_candidates),
            "warnings": [
                "Descriptive school-level association only; no individual linkage or causal inference.",
                "Historic admissions/accepted and unique applicant adjustment are unavailable in source files.",
                "Registrations are programme-row applications and may include multiple applications per student.",
                "M component can have low JPZ spread in some cohorts; residual ranking can be unstable.",
                "Scope limitation: expected-vs-observed modeling uses GY4/GY6/GY8 and defensible 4-year SOS rows (entrant grade 9 only); SOS matching is school × SMO16-category (not KKOV/programme-level).",
            ],
            "components": model_meta.get("components", {}),
        },
    )


def _cross_year_descriptive(jpz_components: pd.DataFrame, mz_components: pd.DataFrame, jpz_modern: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    jpz = jpz_components[jpz_components["component"].isin(["CJ", "M"])].copy() if not jpz_components.empty else pd.DataFrame()
    if not jpz.empty:
        jpz = jpz[~jpz["identity_quality"].eq("unresolved")].copy()
    if jpz.empty:
        pd.DataFrame(columns=["school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode", "school_type", "programme_taxonomy", "programme_identity", "classification_quality", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component", "mean_jpz_percentile", "n_years", "first_year", "last_year", "slope_per_year"]).to_csv(
            out_dir / "jpz_school_component_trends.csv", index=False
        )
    else:
        jpz_grp = (
            jpz.groupby([
                "school_key",
                "identity_quality",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "programme_group_raw",
                "programme_group_16_raw",
                "kkov_raw",
                "programme_name_raw",
                "programme_focus_raw",
                "component",
            ], dropna=False)
            .agg(
                mean_jpz_percentile=("mean_percentile", "mean"),
                n_years=("entry_year", "nunique"),
                first_year=("entry_year", "min"),
                last_year=("entry_year", "max"),
            )
            .reset_index()
        )
        slopes = (
            jpz.groupby(["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], dropna=False)
            .apply(lambda g: _linear_slope(g["entry_year"], g["mean_percentile"]), include_groups=False)
            .reset_index(name="slope_per_year")
        )
        jpz_grp = jpz_grp.merge(slopes, on=["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], how="left")
        jpz_grp.to_csv(out_dir / "jpz_school_component_trends.csv", index=False)

    if mz_components.empty:
        pd.DataFrame(columns=["school_key", "identity_quality", "redizo", "school_name_raw", "address_raw", "city", "postcode", "school_type", "programme_taxonomy", "programme_identity", "classification_quality", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component", "mean_mz_score_pct", "mean_mz_candidates", "max_mz_candidates", "mz_school_mean_percentile", "mz_school_mean_percentile_method", "mz_school_mean_percentile_reference", "mean_mz_participation_rate_vs_cj", "n_years", "first_year", "last_year", "slope_per_year", "slope_mz_mean_score_pct_per_year", "slope_mz_school_mean_percentile_per_year"]).to_csv(
            out_dir / "mz_school_component_trends.csv", index=False
        )
    else:
        mz_pref = (
            mz_components[~mz_components["identity_quality"].eq("unresolved")]
            .assign(_rank=lambda d: d["variant"].map({"jap": 0, "j": 1}).fillna(9))
            .sort_values(["year", "school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component", "_rank"])
            .drop_duplicates(["year", "school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], keep="first")
        )
        mz_pref = _attach_mz_school_mean_percentile(mz_pref)
        mz_pref = mz_pref.merge(
            _compute_mz_participation_rate_vs_cj(mz_pref.rename(columns={"school_type": "match_school_type"})).rename(
                columns={"match_school_type": "school_type"}
            ),
            on=["school_key", "school_type", "year", "component"],
            how="left",
        )
        mz_grp = (
            mz_pref.groupby([
                "school_key",
                "identity_quality",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "school_type",
                "programme_taxonomy",
                "programme_identity",
                "classification_quality",
                "programme_group_raw",
                "programme_group_16_raw",
                "kkov_raw",
                "programme_name_raw",
                "programme_focus_raw",
                "component",
            ], dropna=False)
            .agg(
                mean_mz_score_pct=("mean_score", "mean"),
                mean_mz_candidates=("candidates", "mean"),
                max_mz_candidates=("candidates", "max"),
                mz_school_mean_percentile=("mz_school_mean_percentile", "mean"),
                mz_school_mean_percentile_method=("mz_school_mean_percentile_method", "first"),
                mz_school_mean_percentile_reference=("mz_school_mean_percentile_reference", "first"),
                mean_mz_participation_rate_vs_cj=("mz_participation_rate_vs_cj", "mean"),
                n_years=("year", "nunique"),
                first_year=("year", "min"),
                last_year=("year", "max"),
            )
            .reset_index()
        )
        mz_slopes = (

            mz_pref.groupby(["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], dropna=False)
            .apply(lambda g: _linear_slope(g["year"], g["mean_score"]), include_groups=False)
            .reset_index(name="slope_per_year")
        )
        mz_score_slopes = (
            mz_pref.groupby(["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], dropna=False)
            .apply(lambda g: _linear_slope(g["year"], g["mean_score"]), include_groups=False)
            .reset_index(name="slope_mz_mean_score_pct_per_year")
        )
        mz_pct_slopes = (
            mz_pref.groupby(["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], dropna=False)
            .apply(lambda g: _linear_slope(g["year"], g["mz_school_mean_percentile"]), include_groups=False)
            .reset_index(name="slope_mz_school_mean_percentile_per_year")
        )
        mz_grp = (
            mz_grp.merge(mz_slopes, on=["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], how="left")
            .merge(mz_score_slopes, on=["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], how="left")
            .merge(mz_pct_slopes, on=["school_key", "school_type", "programme_identity", "programme_group_raw", "programme_group_16_raw", "kkov_raw", "programme_name_raw", "programme_focus_raw", "component"], how="left")
        )
        mz_grp.to_csv(out_dir / "mz_school_component_trends.csv", index=False)

    school_dim = _build_school_dimension(jpz_components, jpz_modern, mz_components)
    unified = _build_unified_school_history(school_dim, jpz_components, mz_components, jpz_modern)
    unified.to_csv(out_dir / "school_history_unified.csv", index=False)

    dump_json(
        out_dir / "metadata.json",
        {
            "analysis_type": "cross_year_descriptive_only",
            "cohort_pairing": "none",
            "note": "JPZ and MZ are reported as separate annual school-component panels; no pseudo-cohort linkage is inferred.",
        },
    )


def analyze_archive_local(archive_dir: str | Path) -> dict[str, str]:
    arch = Path(archive_dir)
    manifest_path = arch / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Archive manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    sources = manifest.get("sources", [])
    required_missing = [s for s in sources if s.get("required") and s.get("status") != "downloaded"]
    if required_missing:
        missing_ids = [str(s.get("source_id")) for s in required_missing]
        raise FileNotFoundError(f"Required local sources are missing/unavailable in archive: {missing_ids}")

    raw_by_id: dict[str, Path] = {}
    for s in sources:
        if s.get("status") != "downloaded":
            continue
        rel = s.get("raw_file")
        sid = s.get("source_id")
        if rel and sid:
            p = arch / str(rel)
            if p.exists():
                raw_by_id[str(sid)] = p

    normalized_dir = ensure_dir(arch / "normalized")
    reports_dir = ensure_dir(arch / "reports")

    parse_diagnostics: list[dict[str, Any]] = []

    jpz_hist_frames: list[pd.DataFrame] = []
    for y in range(2017, 2024):
        sid = f"jpz_{y}_historic_vysledky"
        p = raw_by_id.get(sid)
        if p is None:
            parse_diagnostics.append({"dataset": "jpz_historic", "source_id": sid, "status": "absent", "reason": "not_downloaded"})
            continue
        try:
            parsed = parse_jpz_historic_components(p, y, sid)
            parse_diagnostics.append(
                {
                    "dataset": "jpz_historic",
                    "source_id": sid,
                    "status": "ok" if not parsed.empty else "empty",
                    "rows": int(len(parsed)),
                    "components": {k: int(v) for k, v in parsed.get("component", pd.Series(dtype="string")).value_counts(dropna=False).to_dict().items()} if not parsed.empty else {},
                }
            )
            if not parsed.empty:
                jpz_hist_frames.append(parsed)
        except Exception as exc:
            parse_diagnostics.append({"dataset": "jpz_historic", "source_id": sid, "status": "parser_error", "error": str(exc)})

    jpz_components = pd.concat(jpz_hist_frames, ignore_index=True) if jpz_hist_frames else pd.DataFrame()

    modern_frames: list[pd.DataFrame] = []
    for y in range(2024, 2027):
        app_sid = f"jpz_{y}_kolo1_prihlasky"
        cap_sid = f"jpz_{y}_kolo1_kapacity"
        res_sid = f"jpz_{y}_kolo1_vysledky"
        app = raw_by_id.get(app_sid)
        cap = raw_by_id.get(cap_sid)
        res = raw_by_id.get(res_sid)
        if app is None or cap is None:
            parse_diagnostics.append(
                {
                    "dataset": "jpz_modern_triplet",
                    "year": y,
                    "source_id": f"jpz_{y}_kolo1_triplet",
                    "status": "absent",
                    "reason": "required_triplet_files_missing",
                    "app_present": bool(app is not None),
                    "cap_present": bool(cap is not None),
                }
            )
            continue
        try:
            parsed = parse_jpz_modern_triplet(app, cap, res, y, app_sid, cap_sid, res_sid if res is not None else None)
            parse_diagnostics.append(
                {
                    "dataset": "jpz_modern_triplet",
                    "year": y,
                    "source_id": f"jpz_{y}_kolo1_triplet",
                    "status": "ok" if not parsed.empty else "empty",
                    "rows": int(len(parsed)),
                }
            )
            if not parsed.empty:
                modern_frames.append(parsed)
        except Exception as exc:
            parse_diagnostics.append(
                {
                    "dataset": "jpz_modern_triplet",
                    "year": y,
                    "source_id": f"jpz_{y}_kolo1_triplet",
                    "status": "parser_error",
                    "error": str(exc),
                }
            )

    jpz_modern = pd.concat(modern_frames, ignore_index=True) if modern_frames else pd.DataFrame()

    mz_frames: list[pd.DataFrame] = []
    mz_downloaded = sorted([sid for sid in raw_by_id if sid.startswith("mz_")])
    for sid in mz_downloaded:
        m = re.fullmatch(r"mz_(\d{4})_(j|jap)", sid)
        if not m:
            continue
        y = int(m.group(1))
        v = m.group(2)
        p = raw_by_id[sid]
        try:
            parsed = parse_mz_components(p, y, v, sid)
            comp_counts = {k: int(vv) for k, vv in parsed.get("component", pd.Series(dtype="string")).value_counts(dropna=False).to_dict().items()} if not parsed.empty else {}
            parse_diagnostics.append(
                {
                    "dataset": "mz",
                    "source_id": sid,
                    "year": y,
                    "variant": v,
                    "status": "ok" if not parsed.empty else "empty",
                    "rows": int(len(parsed)),
                    "components": comp_counts,
                }
            )
            if not parsed.empty:
                mz_frames.append(parsed)
        except Exception as exc:
            parse_diagnostics.append(
                {
                    "dataset": "mz",
                    "source_id": sid,
                    "year": y,
                    "variant": v,
                    "status": "parser_error",
                    "error": str(exc),
                }
            )

    mz_components = pd.concat(mz_frames, ignore_index=True) if mz_frames else pd.DataFrame()

    diagnostics_path = reports_dir / "parser_diagnostics.json"
    dump_json(diagnostics_path, {"archive": str(arch), "diagnostics": parse_diagnostics})

    historic_jpz_years = sorted(
        {
            int(m.group(1))
            for sid in raw_by_id
            for m in [re.fullmatch(r"jpz_(\d{4})_historic_vysledky", sid)]
            if m is not None
        }
    )
    required_grad_years = sorted({y + 8 for y in historic_jpz_years if (y + 8) <= 2026})
    mz_sources_by_year: dict[int, list[dict[str, Any]]] = {}
    for d in parse_diagnostics:
        if d.get("dataset") != "mz":
            continue
        y = d.get("year")
        if y is None:
            continue
        mz_sources_by_year.setdefault(int(y), []).append(d)

    absent_grad_years = [y for y in required_grad_years if y not in mz_sources_by_year]
    if absent_grad_years:
        raise FileNotFoundError(
            f"Required MZ cohort graduation years are absent from local archive sources: {absent_grad_years}. Diagnostics: {diagnostics_path}"
        )

    malformed_grad_years: list[int] = []
    no_gy8_grad_years: list[int] = []
    for gy in required_grad_years:
        entries = mz_sources_by_year.get(gy, [])
        if not entries:
            continue
        has_parser_error = any(e.get("status") == "parser_error" for e in entries)
        parsed_with_rows = [e for e in entries if e.get("status") == "ok" and int(e.get("rows", 0)) > 0]
        parsed_with_cjm = [
            e
            for e in parsed_with_rows
            if int((e.get("components") or {}).get("CJ", 0)) > 0 and int((e.get("components") or {}).get("M", 0)) > 0
        ]
        if has_parser_error and not parsed_with_rows:
            malformed_grad_years.append(gy)
        elif not parsed_with_cjm:
            no_gy8_grad_years.append(gy)

    if malformed_grad_years:
        raise ValueError(
            f"Required MZ cohort graduation years failed to parse (malformed sources): {malformed_grad_years}. Diagnostics: {diagnostics_path}"
        )
    if no_gy8_grad_years:
        raise ValueError(
            f"Required MZ cohort graduation years parsed but yielded no gymnasium CJ/M components: {no_gy8_grad_years}. Diagnostics: {diagnostics_path}"
        )

    if not mz_downloaded:
        raise FileNotFoundError(
            f"No MZ sources are present in archive manifest/downloads; expected downloaded mz_<year>_<variant> files. Diagnostics: {diagnostics_path}"
        )
    if mz_components.empty:
        had_parser_error = any(d.get("dataset") == "mz" and d.get("status") == "parser_error" for d in parse_diagnostics)
        if had_parser_error:
            raise ValueError(
                f"All available MZ sources failed to parse (malformed layout/content) and yielded no gymnasium components. Diagnostics: {diagnostics_path}"
            )
        raise ValueError(
            f"Available MZ sources parsed but yielded no gymnasium components (redizo_smo16). Diagnostics: {diagnostics_path}"
        )

    mz_cj_m = mz_components[mz_components["component"].isin(["CJ", "M"])]
    if mz_cj_m.empty:
        raise ValueError(
            f"MZ parsing produced rows but no CJ/M component facts for gymnasium rows, so cohort analysis cannot proceed. Diagnostics: {diagnostics_path}"
        )

    if jpz_components.empty:
        jpz_components = pd.DataFrame(
            columns=[
                "entry_year",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "programme_group_raw",
                "grade",
                "school_type",
                "programme_duration_years",
                "classification_quality",
                "component",
                "mean_percentile",
                "sd",
                "registered",
                "sat",
                "admitted",
                "metric_name",
                "source_id",
                "source_row_number",
            ]
        )
    if jpz_modern.empty:
        jpz_modern = pd.DataFrame(
            columns=[
                "redizo",
                "school_name_raw",
                "address_raw",
                "programme_group_raw",
                "grade",
                "school_type",
                "programme_duration_years",
                "classification_quality",
                "applications",
                "capacity",
                "entry_year",
                "city",
                "postcode",
                "actual_score_field",
                "actual_score_value",
                "actual_score_unit",
            ]
        )
    if mz_components.empty:
        mz_components = pd.DataFrame(
            columns=[
                "year",
                "variant",
                "redizo",
                "school_name_raw",
                "address_raw",
                "city",
                "postcode",
                "programme_group_raw",
                "grade",
                "school_type",
                "programme_duration_years",
                "classification_quality",
                "component",
                "candidates",
                "mean_score",
                "pass_rate",
                "source_id",
                "source_row_number",
            ]
        )

    school_dim = _build_school_dimension(jpz_components, jpz_modern, mz_components)
    jpz_components = _attach_school_identity(jpz_components, school_dim)
    jpz_modern = _attach_school_identity(jpz_modern, school_dim)
    mz_components = _attach_school_identity(mz_components, school_dim)

    jpz_components.to_csv(normalized_dir / "jpz_components.csv", index=False)
    jpz_modern.to_csv(normalized_dir / "jpz_modern_round1.csv", index=False)
    mz_components.to_csv(normalized_dir / "mz_components.csv", index=False)
    school_dim.to_csv(normalized_dir / "school_dimension.csv", index=False)

    _cohort_matched(jpz_components, mz_components, ensure_dir(reports_dir / "cohort_matched"))
    cohort_panel = pd.read_csv(reports_dir / "cohort_matched" / "cohort_component_panel.csv")
    _expected_vs_observed_association(cohort_panel, ensure_dir(reports_dir / "cohort_matched"))
    _cross_year_descriptive(jpz_components, mz_components, jpz_modern, ensure_dir(reports_dir / "cross_year_descriptive"))

    methodology = {
        "language": "en",
        "metric_units": {
            "JPZ": "percentile",
            "MZ": "percentage score / pass-rate proportion",
        },
        "archive_availability": {
            "explicit_unavailable": [s for s in sources if s.get("status") == "unavailable"],
        },
        "notes": [
            "All analytics are computed strictly from local archived files.",
            "Cohort-matched and cross-year descriptive outputs are separated by design.",
            "No invalid synthetic CELKEM JPZ metric is created.",
            "School type classification uses only official source programme/grade coding fields (programme_group_raw, grade) and preserves unknown classifications; CELKEM/TOTAL_AGGREGATE rows are excluded from program-level outputs.",
            "Cohort matching and expected-vs-observed modeling use GY4/GY6/GY8 plus defensible SOS types with duration=4 and entrant grade=9; for SOS, matching granularity is school × SMO16-category and not KKOV/programme-level.",
            "Historic JPZ mean_percentile is a published test-result aggregate; it is not admissions acceptance.",
            "Historic PŘIHLÁŠENI/KONALI are programme-row registrations/test-takers, not unique applicants or enrolment.",
            "Historic accepted/admitted counts are unavailable in these sources and remain null.",
            "Modern 2024+ JPZ triplet metrics are a separate metric family and are not compared numerically to historic JPZ mean percentiles.",
            "Synthetic intake selectivity is scenario-based and non-identifying: it does not reconstruct school-to-school preferences, accepted counts, acceptance rates, or causal effects.",
            "Scenario intake selectivity uses CJ+M z-space input and an MZ-throughput capacity proxy (max candidates across CJ/M/AJ in matched graduation-year school_type); when proxy/test-taker inputs are unavailable the synthetic metric is null.",
            "AJ appears only as descriptive MZ outcome rows joined to the same CJ+M synthetic input entity; no AJ JPZ baseline is inferred.",
        ],
        "diagnostics_file": str(diagnostics_path),
    }
    dump_json(reports_dir / "methodology.json", methodology)

    dashboard_path = write_archive_dashboard(arch)

    return {
        "archive": str(arch),
        "normalized_dir": str(normalized_dir),
        "reports_dir": str(reports_dir),
        "dashboard_html": str(dashboard_path),
    }
