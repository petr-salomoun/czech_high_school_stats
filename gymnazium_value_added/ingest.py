from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
import unicodedata

import pandas as pd

from gymnazium_value_added.archive_pipeline import parse_mz_components
from gymnazium_value_added.config import load_json

LOGGER = logging.getLogger(__name__)

ADMISSIONS_REQUIRED = {
    "school_id",
    "school_name",
    "year",
    "applications",
    "capacity",
}

JPZ_HISTORIC_REQUIRED = {
    "school_id",
    "school_name",
    "year",
    "avg_admission_score",
}

MATURITA_REQUIRED = {
    "school_id",
    "school_name",
    "year",
    "candidates",
}

_SCORE_HEADER_HINTS = {
    "avg_admission_score",
    "mean_score",
    "prumer_jpz_bodu",
    "prumer_bodu",
    "prumerny_bod",
    "prumerny_skor",
    "prumerny_procentni_skor",
    "body_prumer",
    "body_celkem_prumer",
    "prumerne_percentilove_umisteni",
    "prumerny_percentil",
}


def _norm_header(name: str) -> str:
    txt = str(name).strip().lower()
    txt = "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt


def _looks_like_score_header(source_header: str) -> bool:
    return _norm_header(source_header) in _SCORE_HEADER_HINTS


def _extract_year_hint(*values: object) -> int | None:
    for value in values:
        if value is None:
            continue
        m = re.search(r"(19|20)\d{2}", str(value))
        if m:
            return int(m.group(0))
    return None


def _extract_redizo(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    m = re.search(r"\b\d{6,10}\b", text)
    if m:
        return m.group(0)
    return None


def _normalise_avg_score_pair(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    left_s = pd.to_numeric(df[left], errors="coerce")
    right_s = pd.to_numeric(df[right], errors="coerce")
    return pd.concat([left_s, right_s], axis=1).mean(axis=1, skipna=True)


def _historic_jpz_profile(df: pd.DataFrame) -> dict[str, str | None]:
    headers = {_norm_header(h): h for h in df.columns}
    headers_by_norm: dict[str, list[str]] = {}
    for h in df.columns:
        headers_by_norm.setdefault(_norm_header(h), []).append(h)

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in headers:
                return headers[candidate]
        return None

    def pick_all(*candidates: str) -> list[str]:
        picked: list[str] = []
        for candidate in candidates:
            picked.extend(headers_by_norm.get(candidate, []))
            picked.extend([h for norm_name, items in headers_by_norm.items() if norm_name.startswith(candidate) for h in items])
        return picked

    combined = next((headers[k] for k in headers if "redizo" in k and "oborova_skupina" in k), None)
    school_id = pick("red_izo", "redizo", "izo") or combined
    school_name = pick("nazev_skoly", "nazev")
    program_code = pick("oborova_skupina", "obor_kod", "kod_oboru")
    program_name = pick("oborova_skupina", "obor_nazev", "nazev_oboru", "obor")
    avg_score = pick("avg_admission_score", "mean_score", "prumer_jpz_bodu", "prumer_bodu", "prumerny_bod")
    score_pair = pick_all("prumerne_percentilove_umisteni")
    if len(score_pair) < 2:
        score_pair = pick_all("prumerny_percentil", "prumerne_percentilove_umisteni")
    score_pair = list(dict.fromkeys(score_pair))
    czech = pick("cesky_jazyk", "cestina", "cesky_jazyk_test", "cesky_jazyk_hodnoceni")
    math = pick("matematika", "matematika_test", "matematika_hodnoceni")
    rocnik = pick("rocnik", "rocnik_", "rocnik_studium")
    return {
        "school_id": school_id,
        "school_name": school_name,
        "program_code": program_code,
        "program_name": program_name,
        "avg_admission_score": avg_score,
        "avg_score_pair": "|".join(score_pair[:2]) if len(score_pair) >= 2 else None,
        "czech_score": czech,
        "math_score": math,
        "combined_school_id": combined,
        "rocnik": rocnik,
    }


def _pick_existing_source(df: pd.DataFrame, *candidates: str | None) -> str | None:
    for candidate in candidates:
        if candidate and candidate in df.columns:
            return candidate
    return None


def _wide_maturita_profile(df: pd.DataFrame) -> dict[str, str | None]:
    norm_to_raw = {_norm_header(col): col for col in df.columns}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in norm_to_raw:
                return norm_to_raw[cand]
        return None

    return {
        "tier": pick("trideni"),
        "school_id": pick("redizo"),
        "school_name": pick("nazev_skoly"),
        "year": pick("rok"),
        "candidates": pick("konali"),
        "mean_score": pick("prumerny_skor", "prumerny_procentni_skor"),
        "pass_rate": pick("podil_uspesnych"),
    }


def _normalize_maturita_wide_aggregate(df: pd.DataFrame, source_year: int | None, source_id: str, sheet_name: str | int | None) -> pd.DataFrame | None:
    profile = _wide_maturita_profile(df)
    if not all(profile[key] for key in ("tier", "school_id", "school_name", "year", "candidates", "mean_score")):
        return None

    tier = df[profile["tier"]].astype("string").str.strip().str.lower()
    school_id = df[profile["school_id"]].map(_extract_redizo)
    mask = tier.eq("redizo") & school_id.notna()
    if not mask.any():
        return None

    norm = pd.DataFrame(
        {
            "school_id": school_id[mask],
            "school_name": df.loc[mask, profile["school_name"]].astype("string").str.strip(),
            "year": pd.to_numeric(df.loc[mask, profile["year"]], errors="coerce").astype("Int64"),
            "subject": "CELKEM",
            "candidates": pd.to_numeric(df.loc[mask, profile["candidates"]], errors="coerce"),
            "mean_score": pd.to_numeric(df.loc[mask, profile["mean_score"]], errors="coerce"),
        }
    )
    if profile.get("pass_rate"):
        norm["pass_rate"] = pd.to_numeric(df.loc[mask, profile["pass_rate"]], errors="coerce")
        if norm["pass_rate"].notna().any() and float(norm["pass_rate"].max()) > 1.0:
            norm["pass_rate"] = norm["pass_rate"] / 100.0
    else:
        norm["pass_rate"] = pd.NA

    if source_year is not None:
        norm["year"] = norm["year"].fillna(source_year)
    norm = norm.dropna(subset=["school_id", "school_name", "year", "candidates", "mean_score"]).copy()
    if norm.empty:
        return None

    norm["school_id"] = norm["school_id"].astype("string").str.strip()
    norm["school_name"] = norm["school_name"].astype("string").str.strip()
    norm["subject"] = norm["subject"].astype("string")
    norm["candidates"] = pd.to_numeric(norm["candidates"], errors="coerce")
    norm["mean_score"] = pd.to_numeric(norm["mean_score"], errors="coerce")
    norm["pass_rate"] = pd.to_numeric(norm["pass_rate"], errors="coerce")
    norm["source_id"] = source_id
    norm["source_row_number"] = df.index.to_series()[mask].astype(int) + 2
    norm["source_sheet"] = sheet_name if sheet_name is not None else 0
    norm["mapping_profile"] = "maturita_wide_aggregate_redizo"
    return norm.reset_index(drop=True)


def read_table(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet_name if sheet_name is not None else 0)
    return pd.read_csv(p)


def discover_columns(path: str | Path, sheet_name: str | int | None = None, sample_rows: int = 5) -> dict[str, Any]:
    df = read_table(path, sheet_name=sheet_name)
    return {
        "path": str(path),
        "sheet_name": sheet_name,
        "columns": list(df.columns),
        "sample": df.head(sample_rows).to_dict(orient="records"),
    }


def discover_columns_excel(path: str | Path, sample_rows: int = 5) -> dict[str, Any]:
    p = Path(path)
    xls = pd.ExcelFile(p)
    sheets: list[dict[str, Any]] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet)
        sheets.append(
            {
                "sheet_name": sheet,
                "columns": list(df.columns),
                "row_count": int(len(df)),
                "sample": df.head(sample_rows).to_dict(orient="records"),
            }
        )
    return {"path": str(path), "sheets": sheets}


def _validate_mapping(mapping: dict[str, str], required: set[str], mapping_name: str) -> None:
    missing = sorted(required - set(mapping.keys()))
    if missing:
        raise ValueError(f"Mapping '{mapping_name}' is missing required target columns: {', '.join(missing)}")


def _rename_and_project(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    missing_sources = [src for src in mapping.values() if src not in df.columns]
    if missing_sources:
        raise ValueError(f"Mapped source columns are missing in input file: {', '.join(map(str, missing_sources))}")
    projected = df[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})
    return projected


def _auto_map(df: pd.DataFrame, schema: str) -> dict[str, str]:
    headers = list(df.columns)
    norm = {_norm_header(h): h for h in headers}

    def pick(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in norm:
                return norm[c]
        return None

    common = {
        "school_id": pick(["school_id", "red_izo", "izo", "skola_red_izo", "redizo"]),
        "school_name": pick(["school_name", "nazev_skoly", "skola", "nazev"]),
        "year": pick(["year", "rok", "rok_prijimacek", "rok_maturity", "maturitni_rok"]),
        "program_code": pick(["program_code", "kkov", "obor_kod", "kod_oboru"]),
        "program_name": pick(["program_name", "obor_nazev", "nazev_oboru", "obor"]),
    }

    if schema == "admissions":
        mapped = {
            "school_id": common["school_id"],
            "school_name": common["school_name"],
            "year": common["year"],
            "program_code": common["program_code"],
            "program_name": common["program_name"],
            "applications": pick(["applications", "pocet_prihlasek", "prihlasky", "prihlasky_celkem"]),
            "capacity": pick(["capacity", "kapacita", "pocet_mist", "mista"]),
            "avg_admission_score": pick(["avg_admission_score", "prumer_jpz_bodu", "prumer_bodu", "prumerny_bod"]),
        }
    elif schema == "jpz_historic":
        mapped = {
            "school_id": common["school_id"],
            "school_name": common["school_name"],
            "year": common["year"],
            "program_code": common["program_code"],
            "program_name": common["program_name"],
            "avg_admission_score": pick(["avg_admission_score", "prumer_jpz_bodu", "prumer_bodu", "prumerny_bod"]),
        }
    else:
        mapped = {
            "school_id": common["school_id"],
            "school_name": common["school_name"],
            "year": common["year"],
            "subject": pick(["subject", "predmet", "zkouska", "didakticky_test", "cast"]),
            "candidates": pick(["candidates", "pocet_zaku", "maturantu", "pocet_konajicich", "konajici", "k.", "prihlaseni", "konali"]),
            "mean_score": pick(["mean_score", "prumerny_skor", "prumerny_procentni_skor", "prumerny_bod", "prumerny_bodu", "prumer", "body_prumer"]),
            "pass_rate": pick(["pass_rate", "uspesnost", "prospelo", "podil_uspesnych", "miraprospelosti"]),
        }

    return {k: v for k, v in mapped.items() if v is not None}


def _resolve_mapping(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    key: str,
    schema: str,
    required_fields: set[str] | None = None,
    prefer_explicit: bool = True,
) -> tuple[dict[str, str], list[str]]:
    explicit = cfg.get(key, {})
    auto = _auto_map(df, schema)
    merged = {**auto, **explicit} if prefer_explicit else {**explicit, **auto}
    required = required_fields if required_fields is not None else (ADMISSIONS_REQUIRED if schema == "admissions" else MATURITA_REQUIRED)
    missing = sorted(required - set(merged.keys()))
    if missing:
        raise ValueError(
            f"Unable to map required columns for {schema}: {', '.join(missing)}. "
            f"Available source columns: {', '.join(map(str, df.columns))}. "
            "Provide explicit mapping in config/column_mappings.json."
        )
    mapping_log = [f"{k} <- {v}" for k, v in sorted(merged.items())]
    return merged, mapping_log


def _normalize_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["school_id"] = out["school_id"].astype("string").str.strip()
    out["school_name"] = out["school_name"].astype("string").str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def normalize_admissions(
    source_path: str | Path,
    mapping_path: str | Path,
    source_id: str,
    sheet_name: str | int | None = None,
    prefer_explicit: bool = True,
) -> pd.DataFrame:
    cfg = load_json(mapping_path)
    df = read_table(source_path, sheet_name=sheet_name)
    mapping, _ = _resolve_mapping(df, cfg, "admissions_mapping", "admissions", prefer_explicit=prefer_explicit)
    norm = _rename_and_project(df, mapping)
    norm = _normalize_common(norm)

    if "program_code" not in norm.columns:
        norm["program_code"] = None
    if "program_name" not in norm.columns:
        norm["program_name"] = None
    if "avg_admission_score" not in norm.columns:
        norm["avg_admission_score"] = None

    norm["applications"] = pd.to_numeric(norm["applications"], errors="coerce")
    norm["capacity"] = pd.to_numeric(norm["capacity"], errors="coerce")
    if "avg_admission_score" in mapping and _looks_like_score_header(mapping["avg_admission_score"]):
        norm["avg_admission_score"] = pd.to_numeric(norm["avg_admission_score"], errors="coerce")
    else:
        norm["avg_admission_score"] = pd.NA

    norm["source_id"] = source_id
    norm["source_row_number"] = pd.RangeIndex(start=2, stop=len(norm) + 2)
    norm["mapping_profile"] = json_safe(mapping)
    return norm


def normalize_jpz_historic_results(
    source_path: str | Path,
    mapping_path: str | Path,
    source_id: str,
    sheet_name: str | int | None = None,
    prefer_explicit: bool = True,
) -> pd.DataFrame:
    cfg = load_json(mapping_path)
    p = Path(source_path)
    source_year = _extract_year_hint(source_id, p.name, p.stem)
    header_row: int | None = None
    candidate_headers = tuple(range(0, 6))

    def _read_candidate_sheet(sheet: str | int) -> tuple[pd.DataFrame, dict[str, str | None], int] | None:
        nonlocal header_row
        for candidate_header in candidate_headers:
            df_try = pd.read_excel(p, sheet_name=sheet, header=candidate_header)
            profile_try = _historic_jpz_profile(df_try)
            if profile_try.get("school_id") and profile_try.get("school_name") and (
                profile_try.get("avg_admission_score")
                or profile_try.get("avg_score_pair")
                or (profile_try.get("czech_score") and profile_try.get("math_score"))
            ):
                header_row = candidate_header
                return df_try, profile_try, candidate_header
        return None

    if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"} and sheet_name is None:
        xls = pd.ExcelFile(p)
        chosen_df: pd.DataFrame | None = None
        chosen_profile: dict[str, str | None] | None = None
        for sheet in xls.sheet_names:
            candidate = _read_candidate_sheet(sheet)
            if candidate is None:
                continue
            chosen_df, chosen_profile, _ = candidate
            sheet_name = sheet
            break
        if chosen_df is None or chosen_profile is None:
            profile = discover_columns_excel(p)
            raise ValueError(
                "Could not identify a compatible historic JPZ results sheet automatically. "
                f"Source profile: {profile}. Provide explicit mapping and --maturita-sheet/worksheet equivalent."
            )
        df = chosen_df
        profile = chosen_profile
    else:
        if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
            candidate = _read_candidate_sheet(sheet_name if sheet_name is not None else 0)
            if candidate is None:
                df = read_table(source_path, sheet_name=sheet_name)
                profile = _historic_jpz_profile(df)
            else:
                df, profile, _ = candidate
        else:
            df = read_table(source_path, sheet_name=sheet_name)
            profile = _historic_jpz_profile(df)

    explicit_mapping = cfg.get("admissions_mapping", {}) if prefer_explicit else {}

    school_id_src = _pick_existing_source(df, explicit_mapping.get("school_id"), profile.get("school_id"))
    school_name_src = _pick_existing_source(df, explicit_mapping.get("school_name"), profile.get("school_name"))
    if not school_id_src or not school_name_src:
        raise ValueError(
            "Historic JPZ results workbook is missing clearly identifiable school and program columns."
        )

    program_code_src = _pick_existing_source(df, explicit_mapping.get("program_code"), profile.get("program_code"), profile.get("program_name"))
    program_name_src = _pick_existing_source(df, explicit_mapping.get("program_name"), profile.get("program_name"), program_code_src)
    avg_score_src = _pick_existing_source(df, explicit_mapping.get("avg_admission_score"), profile.get("avg_admission_score"))
    avg_score_pair = str(profile.get("avg_score_pair") or "")
    avg_score_pair_sources = [s for s in avg_score_pair.split("|") if s]

    if source_year is None:
        raise ValueError(
            "Historic JPZ results entry year must come from the source run context or workbook name."
        )

    school_id = df[school_id_src].map(_extract_redizo)
    school_id = school_id.where(school_id.notna(), pd.NA)
    if profile.get("combined_school_id") and school_id.isna().all():
        school_id = df[profile["combined_school_id"]].map(_extract_redizo)
        school_id = school_id.where(school_id.notna(), pd.NA)
    school_name = df[school_name_src].astype("string").str.strip()

    norm = pd.DataFrame({"school_id": school_id, "school_name": school_name})
    norm["year"] = source_year

    if program_code_src and program_code_src in df.columns:
        norm["program_code"] = df[program_code_src].astype("string").str.strip()
    else:
        norm["program_code"] = None
    if program_name_src and program_name_src in df.columns:
        norm["program_name"] = df[program_name_src].astype("string").str.strip()
    else:
        norm["program_name"] = norm["program_code"]

    if avg_score_src and avg_score_src in df.columns:
        norm["avg_admission_score"] = pd.to_numeric(df[avg_score_src], errors="coerce")
        avg_score_method = f"source:{avg_score_src}"
    elif len(avg_score_pair_sources) >= 2 and all(src in df.columns for src in avg_score_pair_sources[:2]):
        norm["avg_admission_score"] = _normalise_avg_score_pair(df, avg_score_pair_sources[0], avg_score_pair_sources[1])
        avg_score_method = f"mean({avg_score_pair_sources[0]},{avg_score_pair_sources[1]})"
    elif profile.get("czech_score") and profile.get("math_score") and profile["czech_score"] in df.columns and profile["math_score"] in df.columns:
        norm["avg_admission_score"] = _normalise_avg_score_pair(df, profile["czech_score"], profile["math_score"])
        avg_score_method = f"mean({profile['czech_score']},{profile['math_score']})"
    else:
        raise ValueError("Historic JPZ results workbook does not contain recognized Czech/Math score headers or a usable score column.")

    norm = _normalize_common(norm)
    norm = norm[norm["school_id"].notna() & norm["school_name"].notna()].copy()
    norm = norm[norm["avg_admission_score"].notna()].copy()
    if norm.empty:
        raise ValueError("Historic JPZ results workbook does not contain valid entrance score/result data.")
    norm["applications"] = pd.NA
    norm["capacity"] = pd.NA
    norm["selectivity_ratio"] = pd.NA
    norm["selection_metric"] = pd.NA
    norm["selection_metric_observed"] = False
    norm["selection_metric_missing"] = True
    norm["selection_metric_method"] = "historic_jpz_results_only"
    norm["avg_admission_score_method"] = avg_score_method
    if profile.get("rocnik") and profile["rocnik"] in df.columns:
        norm["rocnik"] = pd.to_numeric(df[profile["rocnik"]], errors="coerce")
    norm["source_id"] = source_id
    norm["source_row_number"] = pd.RangeIndex(start=(header_row + 2 if header_row is not None else 2), stop=(header_row + 2 if header_row is not None else 2) + len(norm))
    norm["source_sheet"] = sheet_name if sheet_name is not None else 0
    norm["mapping_profile"] = json_safe({
        "school_id": school_id_src,
        "school_name": school_name_src,
        "year": str(source_year),
        "program_code": str(program_code_src) if program_code_src else "",
        "program_name": str(program_name_src) if program_name_src else "",
        "avg_admission_score": str(avg_score_src) if avg_score_src else avg_score_method,
    })
    return norm


def normalize_maturita(
    source_path: str | Path,
    mapping_path: str | Path,
    source_id: str,
    sheet_name: str | int | None = None,
    prefer_explicit: bool = True,
) -> pd.DataFrame:
    cfg = load_json(mapping_path)
    p = Path(source_path)
    source_year = _extract_year_hint(source_id, p.name, p.stem)

    if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        m = re.search(r"(?:^|_)(jap|j)(?:$|[^a-z])", str(source_id).lower()) or re.search(r"(?:^|_)(jap|j)(?:$|[^a-z])", p.stem.lower())
        variant = m.group(1) if m else "j"
        sheet = sheet_name
        if sheet is None:
            try:
                xls = pd.ExcelFile(p)
                sheet = xls.sheet_names[0] if xls.sheet_names else 0
            except Exception:
                sheet = 0
        try:
            parsed = parse_mz_components(p, source_year or 0, variant, source_id)
        except Exception:
            parsed = pd.DataFrame()
        if not parsed.empty:
            norm = pd.DataFrame(
                {
                    "school_id": parsed["redizo"].astype("string"),
                    "school_name": parsed["school_name_raw"].astype("string"),
                    "year": pd.to_numeric(parsed["year"], errors="coerce").astype("Int64"),
                    "subject": parsed["component"].astype("string"),
                    "candidates": pd.to_numeric(parsed["candidates"], errors="coerce"),
                    "mean_score": pd.to_numeric(parsed["mean_score"], errors="coerce"),
                    "pass_rate": pd.to_numeric(parsed["pass_rate"], errors="coerce"),
                    "source_id": source_id,
                    "source_row_number": pd.to_numeric(parsed["source_row_number"], errors="coerce").astype("Int64"),
                    "source_sheet": sheet_name if sheet_name is not None else 0,
                    "mapping_profile": "maturita_mz_components_redizo_smo16",
                }
            )
            return norm.reset_index(drop=True)

    def _try_maturita_mapping(df_try: pd.DataFrame, preferred: bool) -> dict[str, str] | None:
        try:
            mapping_try, _ = _resolve_mapping(df_try, cfg, "maturita_mapping", "maturita", prefer_explicit=preferred)
            _rename_and_project(df_try, mapping_try)
            return mapping_try
        except ValueError:
            return None

    def _read_candidate_sheet(sheet: str | int) -> tuple[pd.DataFrame, dict[str, str], int] | None:
        for candidate_header in (1, 0):
            df_try = pd.read_excel(p, sheet_name=sheet, header=candidate_header)
            mapping_try = _try_maturita_mapping(df_try, prefer_explicit)
            if mapping_try is None and prefer_explicit:
                mapping_try = _try_maturita_mapping(df_try, False)
            if mapping_try is not None:
                return df_try, mapping_try, candidate_header
        return None

    if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"} and sheet_name is None:
        xls = pd.ExcelFile(p)
        for sheet in xls.sheet_names:
            for candidate_header in (1, 0):
                try:
                    wide_df = pd.read_excel(p, sheet_name=sheet, header=candidate_header)
                except Exception:
                    continue
                wide = _normalize_maturita_wide_aggregate(wide_df, source_year, source_id, sheet)
                if wide is not None:
                    return wide
        chosen_df: pd.DataFrame | None = None
        chosen_mapping: dict[str, str] | None = None
        for sheet in xls.sheet_names:
            candidate = _read_candidate_sheet(sheet)
            if candidate is None:
                continue
            chosen_df, chosen_mapping, _ = candidate
            sheet_name = sheet
            break
        if chosen_df is None or chosen_mapping is None:
            if len(xls.sheet_names) == 1:
                single_df = pd.read_excel(p, sheet_name=xls.sheet_names[0], header=1)
                wide = _normalize_maturita_wide_aggregate(single_df, source_year, source_id, xls.sheet_names[0])
                if wide is not None:
                    return wide
            profile = discover_columns_excel(p)
            raise ValueError(
                "Could not identify a compatible maturita sheet automatically. "
                f"Source profile: {profile}. Provide explicit mapping and --maturita-sheet."
            )
        df = chosen_df
        mapping = chosen_mapping
    else:
        if p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
            for candidate_header in (1, 0):
                try:
                    wide_df = pd.read_excel(p, sheet_name=sheet_name if sheet_name is not None else 0, header=candidate_header)
                except Exception:
                    continue
                wide = _normalize_maturita_wide_aggregate(wide_df, source_year, source_id, sheet_name if sheet_name is not None else 0)
                if wide is not None:
                    return wide
            candidate = _read_candidate_sheet(sheet_name if sheet_name is not None else 0)
            if candidate is None:
                df = read_table(source_path, sheet_name=sheet_name)
                mapping = _try_maturita_mapping(df, prefer_explicit)
                if mapping is None and prefer_explicit:
                    mapping = _try_maturita_mapping(df, False)
                if mapping is None:
                    profile = discover_columns_excel(p)
                    raise ValueError(
                        "Could not identify a compatible maturita sheet automatically. "
                        f"Source profile: {profile}. Provide explicit mapping and --maturita-sheet."
                    )
            else:
                df, mapping, _ = candidate
        else:
            df = read_table(source_path, sheet_name=sheet_name)
            mapping = _try_maturita_mapping(df, prefer_explicit)
            if mapping is None and prefer_explicit:
                mapping = _try_maturita_mapping(df, False)
            if mapping is None:
                raise ValueError("Could not identify a compatible maturita sheet automatically.")

    if sheet_name is not None and p.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        wide = _normalize_maturita_wide_aggregate(df, source_year, source_id, sheet_name)
        if wide is not None and (not set(mapping.keys()).issuperset({"school_id", "school_name", "year", "candidates"}) or "mean_score" not in mapping):
            return wide

    norm = _rename_and_project(df, mapping)
    norm = _normalize_common(norm)

    if "subject" not in norm.columns:
        norm["subject"] = "ALL"
    if "mean_score" not in norm.columns:
        norm["mean_score"] = None
    if "pass_rate" not in norm.columns:
        norm["pass_rate"] = None

    norm["subject"] = norm["subject"].astype(str).str.strip()
    norm["candidates"] = pd.to_numeric(norm["candidates"], errors="coerce")
    if "mean_score" in mapping and _looks_like_score_header(mapping["mean_score"]):
        norm["mean_score"] = pd.to_numeric(norm["mean_score"], errors="coerce")
    else:
        norm["mean_score"] = pd.NA
    norm["pass_rate"] = pd.to_numeric(norm["pass_rate"], errors="coerce")
    if norm["pass_rate"].notna().any() and float(norm["pass_rate"].max()) > 1.0:
        norm["pass_rate"] = norm["pass_rate"] / 100.0

    norm["source_id"] = source_id
    norm["source_row_number"] = pd.RangeIndex(start=2, stop=len(norm) + 2)
    norm["source_sheet"] = sheet_name if sheet_name is not None else 0
    norm["mapping_profile"] = json_safe(mapping)
    return norm


def normalize_jpz_triplet(
    applications_path: str | Path,
    capacities_path: str | Path,
    results_path: str | Path | None,
    mapping_path: str | Path,
    source_id_prefix: str,
    sheet_name: str | int | None = None,
    prefer_explicit: bool = True,
) -> pd.DataFrame:
    cfg = load_json(mapping_path)
    app_raw = read_table(applications_path, sheet_name=sheet_name)
    cap_raw = read_table(capacities_path, sheet_name=sheet_name)

    app_map, _ = _resolve_mapping(
        app_raw,
        cfg,
        "admissions_mapping",
        "admissions",
        required_fields={"school_id", "school_name", "year", "applications"},
        prefer_explicit=prefer_explicit,
    )
    cap_map, _ = _resolve_mapping(
        cap_raw,
        cfg,
        "admissions_mapping",
        "admissions",
        required_fields={"school_id", "school_name", "year", "capacity"},
        prefer_explicit=prefer_explicit,
    )

    app_required = ["school_id", "school_name", "year", "program_code", "program_name", "applications"]
    cap_required = ["school_id", "school_name", "year", "program_code", "program_name", "capacity"]

    missing_app = [k for k in app_required if k not in app_map]
    missing_cap = [k for k in cap_required if k not in cap_map]
    if missing_app:
        raise ValueError(
            f"JPZ applications file is missing required mapped fields: {missing_app}. "
            f"Columns available: {list(app_raw.columns)}"
        )
    if missing_cap:
        raise ValueError(
            f"JPZ capacities file is missing required mapped fields: {missing_cap}. "
            f"Columns available: {list(cap_raw.columns)}"
        )

    app_proj = _rename_and_project(app_raw, {k: app_map[k] for k in app_required})
    cap_proj = _rename_and_project(cap_raw, {k: cap_map[k] for k in cap_required})
    app = _normalize_common(app_proj)
    cap = _normalize_common(cap_proj)
    app["applications"] = pd.to_numeric(app["applications"], errors="coerce")
    cap["capacity"] = pd.to_numeric(cap["capacity"], errors="coerce")

    keys = ["school_id", "school_name", "year", "program_code", "program_name"]
    app_agg = app.groupby(keys, dropna=False)["applications"].sum().reset_index()
    cap_agg = cap.groupby(keys, dropna=False)["capacity"].sum().reset_index()
    merged = app_agg.merge(cap_agg, on=keys, how="outer")

    merged["avg_admission_score"] = pd.NA
    merged["selection_metric"] = merged["applications"] / merged["capacity"]
    merged["selection_metric_observed"] = merged["applications"].notna() & merged["capacity"].notna()
    merged["selection_metric_missing"] = ~merged["selection_metric_observed"].fillna(False)
    merged["selection_metric_method"] = "applications_capacity"
    if results_path is not None:
        res_raw = read_table(results_path, sheet_name=sheet_name)
        res_map, _ = _resolve_mapping(
            res_raw,
            cfg,
            "admissions_mapping",
            "admissions",
            required_fields={"school_id", "school_name", "year"},
            prefer_explicit=prefer_explicit,
        )
        res_required = ["school_id", "school_name", "year", "program_code", "program_name", "avg_admission_score"]
        if all(k in res_map for k in res_required):
            res_proj = _rename_and_project(res_raw, {k: res_map[k] for k in res_required})
            res = _normalize_common(res_proj)
            if _looks_like_score_header(res_map["avg_admission_score"]):
                res["avg_admission_score"] = pd.to_numeric(res["avg_admission_score"], errors="coerce")
                if res["avg_admission_score"].notna().any():
                    res_agg = res.groupby(keys, dropna=False)["avg_admission_score"].mean().reset_index()
                    merged = merged.merge(res_agg, on=keys, how="left", suffixes=("", "_res"))
                    if "avg_admission_score_res" in merged.columns:
                        merged["avg_admission_score"] = merged["avg_admission_score_res"]
                        merged = merged.drop(columns=["avg_admission_score_res"])

    merged["source_id"] = source_id_prefix
    merged["source_row_number"] = pd.RangeIndex(start=2, stop=len(merged) + 2)
    merged["mapping_profile"] = "jpz_triplet_merged"
    return merged


def json_safe(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join([f"{k}<-{v}" for k, v in sorted(value.items())])
    return str(value)


def filter_eight_year_programs(
    admissions_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    kkov_codes = set(cfg.get("kkov_codes", ["79-41-K/81"]))
    title_regex = cfg.get("title_regex", r"osmilet")
    rocnik_values = set(cfg.get("rocnik_values", [5]))

    df = admissions_df.copy()
    prog_code = df["program_code"].fillna("").astype(str) if "program_code" in df.columns else pd.Series([""] * len(df), index=df.index)
    prog_name = df["program_name"].fillna("").astype(str) if "program_name" in df.columns else pd.Series([""] * len(df), index=df.index)

    mask_code = prog_code.isin(kkov_codes)
    mask_title = prog_name.str.contains(title_regex, flags=re.IGNORECASE, regex=True, na=False)
    mask_rocnik = pd.Series([False] * len(df), index=df.index)
    if "rocnik" in df.columns and rocnik_values:
        rocnik_series = pd.to_numeric(df["rocnik"], errors="coerce")
        mask_rocnik = rocnik_series.isin(rocnik_values)
    filtered = df[mask_code | mask_title | mask_rocnik]
    LOGGER.info("8-year program filter: %d -> %d rows", len(df), len(filtered))
    return filtered
