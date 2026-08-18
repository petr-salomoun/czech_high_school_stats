from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from gymnazium_value_added.analyze import analyze_value_added
from gymnazium_value_added.archive_pipeline import analyze_archive_local, create_archive
from gymnazium_value_added.archive_report import write_archive_dashboard
from gymnazium_value_added.config import dump_json, ensure_dir, load_json
from gymnazium_value_added.discovery import (
    JPZ_LANDING_URL,
    MZ_LANDING_URL,
    SourceCandidate,
    discover_all_sources,
    select_cohort_pairs,
)
from gymnazium_value_added.ingest import (
    discover_columns_excel,
    discover_columns,
    filter_eight_year_programs,
    normalize_admissions,
    normalize_jpz_historic_results,
    normalize_jpz_triplet,
    normalize_maturita,
)
from gymnazium_value_added.io import DownloadError, download_first_valid_excel, download_url
from gymnazium_value_added.model import AnalysisConfig
from gymnazium_value_added.report import write_report


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        raise ValueError("Manifest must contain a 'sources' list.")
    return manifest


def _candidate_from_dict(obj: dict[str, Any]) -> SourceCandidate:
    year = obj.get("year")
    return SourceCandidate(
        dataset=str(obj["dataset"]),
        year=int(year) if year is not None else None,
        kind=str(obj["kind"]),
        url=str(obj["url"]),
    )


def _load_discovery_override(path: str | Path) -> dict[str, list[SourceCandidate]]:
    data = load_json(path)
    return {
        "jpz": [_candidate_from_dict(x) for x in data.get("jpz", [])],
        "maturita": [_candidate_from_dict(x) for x in data.get("maturita", [])],
    }


def _parse_years(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        values.append(int(t))
    return values or None


def cmd_download(args: argparse.Namespace) -> int:
    manifest = _read_manifest(args.manifest)
    raw_dir = ensure_dir(args.raw_dir)

    download_log: dict[str, Any] = {"downloads": []}
    for src in manifest["sources"]:
        sid = src["id"]
        url = src.get("url", "")
        required = bool(src.get("required", False))
        if not url:
            if required:
                raise ValueError(f"Required source '{sid}' is missing URL.")
            download_log["downloads"].append({"id": sid, "status": "skipped_placeholder"})
            continue

        ext = Path(url).suffix or ".xlsx"
        target = raw_dir / f"{sid}{ext}"
        meta = download_url(url, target, timeout=args.timeout, retries=args.retries, require_excel=args.require_excel)
        expected_sha = src.get("sha256")
        checksum_ok = True
        if expected_sha:
            checksum_ok = str(expected_sha).lower() == str(meta["sha256"]).lower()
            if not checksum_ok:
                raise ValueError(f"Checksum mismatch for {sid}: expected={expected_sha} got={meta['sha256']}")

        download_log["downloads"].append(
            {
                "id": sid,
                "url": url,
                "target": str(target),
                "checksum_ok": checksum_ok,
                **meta,
            }
        )

    out_meta = Path(args.meta_out)
    dump_json(out_meta, download_log)
    return 0


def cmd_discover_columns(args: argparse.Namespace) -> int:
    if str(args.input).lower().endswith((".xlsx", ".xls", ".xlsm")) and args.sheet is None:
        info = discover_columns_excel(args.input)
    else:
        info = discover_columns(args.input, sheet_name=args.sheet)
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return 0


def _resolve_source_file(manifest_path: str | Path, raw_dir: str | Path, source_id: str) -> Path:
    manifest = _read_manifest(manifest_path)
    ids = {s["id"]: s for s in manifest["sources"]}
    if source_id not in ids:
        raise ValueError(f"source_id '{source_id}' is not present in the manifest")
    for p in Path(raw_dir).glob(f"{source_id}.*"):
        return p
    raise FileNotFoundError(f"No file found for source_id={source_id} in {raw_dir}")


def cmd_ingest(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.out_dir)

    admissions_input = Path(args.admissions_input) if args.admissions_input else None
    maturita_input = Path(args.maturita_input) if args.maturita_input else None

    if args.admissions_source_id:
        admissions_input = _resolve_source_file(args.manifest, args.raw_dir, args.admissions_source_id)
    if args.maturita_source_id:
        maturita_input = _resolve_source_file(args.manifest, args.raw_dir, args.maturita_source_id)

    if admissions_input is None or maturita_input is None:
        raise ValueError("Both admissions and maturita inputs are required (file path or source_id).")

    admissions_df = normalize_admissions(
        source_path=admissions_input,
        mapping_path=args.mapping,
        source_id=args.admissions_source_id or admissions_input.name,
        sheet_name=args.admissions_sheet,
    )
    eight_cfg = load_json(args.eight_year_filter) if args.eight_year_filter else {}
    admissions_df = filter_eight_year_programs(admissions_df, config=eight_cfg)

    maturita_df = normalize_maturita(
        source_path=maturita_input,
        mapping_path=args.mapping,
        source_id=args.maturita_source_id or maturita_input.name,
        sheet_name=args.maturita_sheet,
    )

    adm_out = out_dir / "admissions_normalized.csv"
    mat_out = out_dir / "maturita_normalized.csv"
    admissions_df.to_csv(adm_out, index=False, quoting=csv.QUOTE_MINIMAL)
    maturita_df.to_csv(mat_out, index=False, quoting=csv.QUOTE_MINIMAL)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    admissions = pd.read_csv(args.admissions)
    maturita = pd.read_csv(args.maturita)
    cfg_data = load_json(args.analysis_config) if args.analysis_config else {}
    cfg = AnalysisConfig(
        cohort_lag_years=int(cfg_data.get("cohort_lag_years", 8)),
        min_cohort_size=int(cfg_data.get("min_cohort_size", 10)),
        min_school_count=int(cfg_data.get("min_school_count", 10)),
        ridge_alpha=float(cfg_data.get("ridge_alpha", 1.0)),
        bootstrap_iterations=int(cfg_data.get("bootstrap_iterations", 500)),
        random_seed=int(cfg_data.get("random_seed", 42)),
    )
    result_df, meta = analyze_value_added(admissions, maturita, cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    dump_json(args.meta_output, meta)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.analysis_csv)
    methodology = load_json(args.methodology_json)
    write_report(df, methodology, output_dir=args.output_dir, base_name=args.base_name)
    return 0


def _select_urls_for_year(jpz_sources: list[SourceCandidate], year: int) -> dict[str, list[str]]:
    by_kind: dict[str, list[str]] = {"prihlasky": [], "kapacity": [], "vysledky": []}
    for src in jpz_sources:
        if src.year == year and src.kind in by_kind:
            by_kind[src.kind].append(src.url)
    return by_kind


def _select_jpz_result_urls_for_year(jpz_sources: list[SourceCandidate], year: int) -> list[str]:
    return [src.url for src in jpz_sources if src.year == year and src.kind == "vysledky"]


def _select_maturita_urls_for_year(maturita_sources: list[SourceCandidate], year: int) -> list[str]:
    return [src.url for src in maturita_sources if src.year == year]


def cmd_run(args: argparse.Namespace) -> int:
    if args.archive is not None:
        analyze_archive_local(args.archive)
        return 0

    if not args.allow_network_fetch:
        raise ValueError(
            "run no longer performs hidden web fetches. Use 'archive' then 'analyze-local', "
            "or pass --allow-network-fetch to run archive+analyze-local explicitly."
        )

    archive_dir = create_archive(
        archive_root=args.archive_root,
        freeze_id=args.freeze_id,
        year_start=args.year_start,
        year_end=args.year_end,
        refresh=args.refresh,
        timeout=args.timeout,
        retries=args.retries,
        source_override=args.source_override,
    )
    analyze_archive_local(archive_dir)
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    archive_dir = create_archive(
        archive_root=args.archive_root,
        freeze_id=args.freeze_id,
        year_start=args.year_start,
        year_end=args.year_end,
        refresh=args.refresh,
        timeout=args.timeout,
        retries=args.retries,
        source_override=args.source_override,
    )
    print(json.dumps({"archive_dir": str(archive_dir)}, ensure_ascii=False))
    return 0


def cmd_analyze_local(args: argparse.Namespace) -> int:
    result = analyze_archive_local(args.archive)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_report_local(args: argparse.Namespace) -> int:
    dashboard = write_archive_dashboard(args.archive)
    print(json.dumps({"dashboard_html": str(dashboard)}, ensure_ascii=False))
    return 0


def cmd_run_legacy(args: argparse.Namespace) -> int:
    data_dir = ensure_dir(args.data_dir)
    raw_dir = ensure_dir(Path(data_dir) / "raw")
    normalized_dir = ensure_dir(Path(data_dir) / "normalized")
    output_dir = ensure_dir(args.output_dir)

    if args.discovery_json:
        discovered = _load_discovery_override(args.discovery_json)
    else:
        discovered = discover_all_sources(
            jpz_landing_url=args.jpz_landing_url,
            maturita_landing_url=args.maturita_landing_url,
            timeout=args.timeout,
        )

    entry_years = _parse_years(args.entry_years)
    graduation_years = _parse_years(args.graduation_years)
    pairs = select_cohort_pairs(
        discovered["jpz"],
        discovered["maturita"],
        cohort_lag_years=args.cohort_lag,
        entry_years=entry_years,
        graduation_years=graduation_years,
    )
    if not pairs:
        jpz_years = sorted({s.year for s in discovered["jpz"]})
        mz_years = sorted({s.year for s in discovered["maturita"]})
        jpz_kinds = sorted({s.kind for s in discovered["jpz"]})
        raise ValueError(
            "No valid 8-year cohort pair is available from discovered official CERMAT sources. "
            f"JPZ years with files: {jpz_years} (kinds: {jpz_kinds}); Maturita years with files: {mz_years}; cohort_lag={args.cohort_lag}. "
            "Missing data usually means there is no discovered direct JPZ results workbook for the implied entry year, "
            "or no discovered maturita workbook for the matching graduation year."
        )

    selected_pair = max(pairs, key=lambda x: (x["graduation_year"], 1 if x.get("jpz_mode") == "triplet" else 0))
    if args.use_all_pairs:
        selected_pairs = pairs
    else:
        selected_pairs = [selected_pair]

    admissions_frames: list[pd.DataFrame] = []
    maturita_frames: list[pd.DataFrame] = []
    run_meta: dict[str, Any] = {
        "selected_pairs": selected_pairs,
        "downloads": [],
        "discover": {
            "jpz_count": len(discovered["jpz"]),
            "maturita_count": len(discovered["maturita"]),
            "jpz_landing_url": args.jpz_landing_url,
            "maturita_landing_url": args.maturita_landing_url,
        },
    }

    for pair in selected_pairs:
        entry_year = int(pair["entry_year"])
        grad_year = int(pair["graduation_year"])
        jpz_mode = str(pair.get("jpz_mode", "triplet"))
        jpz_urls = _select_urls_for_year(discovered["jpz"], entry_year)
        jpz_result_urls = _select_jpz_result_urls_for_year(discovered["jpz"], entry_year)
        mat_urls = _select_maturita_urls_for_year(discovered["maturita"], grad_year)
        if jpz_mode == "triplet" and (not jpz_urls["prihlasky"] or not jpz_urls["kapacity"]):
            raise ValueError(
                f"Missing required JPZ files for entry year {entry_year}. "
                f"Discovered applications={jpz_urls['prihlasky']}, capacities={jpz_urls['kapacity']}."
            )
        if jpz_mode == "results_only" and not jpz_result_urls:
            raise ValueError(f"Missing historical JPZ results workbook for entry year {entry_year}.")
        if not mat_urls:
            raise ValueError(f"Missing maturita aggregated workbook for graduation year {grad_year}.")

        app_path = raw_dir / f"jpz_{entry_year}_prihlasky.xlsx"
        cap_path = raw_dir / f"jpz_{entry_year}_kapacity.xlsx"
        res_path = raw_dir / f"jpz_{entry_year}_vysledky.xlsx"
        mat_path = raw_dir / f"maturita_{grad_year}.xlsx"

        if jpz_mode == "triplet":
            if args.refresh or not app_path.exists():
                try:
                    used_url, meta = download_first_valid_excel(jpz_urls["prihlasky"], app_path, timeout=args.timeout, retries=args.retries)
                    run_meta["downloads"].append({"type": "jpz_prihlasky", "year": entry_year, "used_url": used_url, **meta})
                except DownloadError as exc:
                    raise RuntimeError(f"Failed to download JPZ applications for year {entry_year}. Attempts: {exc.attempts}") from exc
            if args.refresh or not cap_path.exists():
                try:
                    used_url, meta = download_first_valid_excel(jpz_urls["kapacity"], cap_path, timeout=args.timeout, retries=args.retries)
                    run_meta["downloads"].append({"type": "jpz_kapacity", "year": entry_year, "used_url": used_url, **meta})
                except DownloadError as exc:
                    raise RuntimeError(f"Failed to download JPZ capacities for year {entry_year}. Attempts: {exc.attempts}") from exc
            if jpz_urls["vysledky"] and (args.refresh or not res_path.exists()):
                try:
                    used_url, meta = download_first_valid_excel(jpz_urls["vysledky"], res_path, timeout=args.timeout, retries=args.retries)
                    run_meta["downloads"].append({"type": "jpz_vysledky", "year": entry_year, "used_url": used_url, **meta})
                except DownloadError:
                    res_path = None
        else:
            if args.refresh or not res_path.exists():
                try:
                    used_url, meta = download_first_valid_excel(jpz_result_urls, res_path, timeout=args.timeout, retries=args.retries)
                    run_meta["downloads"].append({"type": "jpz_vysledky_historic", "year": entry_year, "used_url": used_url, **meta})
                except DownloadError as exc:
                    raise RuntimeError(f"Failed to download historical JPZ results for year {entry_year}. Attempts: {exc.attempts}") from exc
        if args.refresh or not mat_path.exists():
            try:
                used_url, meta = download_first_valid_excel(mat_urls, mat_path, timeout=args.timeout, retries=args.retries)
                run_meta["downloads"].append({"type": "maturita", "year": grad_year, "used_url": used_url, **meta})
            except DownloadError as exc:
                raise RuntimeError(f"Failed to download maturita workbook for year {grad_year}. Attempts: {exc.attempts}") from exc

        if jpz_mode == "triplet":
            admissions = normalize_jpz_triplet(
                applications_path=app_path,
                capacities_path=cap_path,
                results_path=res_path if (res_path is not None and Path(res_path).exists()) else None,
                mapping_path=args.mapping,
                source_id_prefix=f"jpz_{entry_year}",
                prefer_explicit=False,
            )
        else:
            admissions = normalize_jpz_historic_results(
                source_path=res_path,
                mapping_path=args.mapping,
                source_id=f"jpz_{entry_year}",
                prefer_explicit=False,
            )
        admissions = filter_eight_year_programs(admissions, config=load_json(args.eight_year_filter))
        admissions_frames.append(admissions)

        maturita = normalize_maturita(
            source_path=mat_path,
            mapping_path=args.mapping,
            source_id=f"maturita_{grad_year}",
            sheet_name=args.maturita_sheet,
            prefer_explicit=False,
        )
        maturita = maturita[maturita["year"].astype("Int64") == grad_year]
        maturita_frames.append(maturita)

    admissions_df = pd.concat(admissions_frames, ignore_index=True)
    maturita_df = pd.concat(maturita_frames, ignore_index=True)

    admissions_csv = normalized_dir / "admissions_normalized.csv"
    maturita_csv = normalized_dir / "maturita_normalized.csv"
    admissions_df.to_csv(admissions_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    maturita_df.to_csv(maturita_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    cfg_data = load_json(args.analysis_config) if args.analysis_config else {}
    cfg = AnalysisConfig(
        cohort_lag_years=int(args.cohort_lag),
        min_cohort_size=int(cfg_data.get("min_cohort_size", 10)),
        min_school_count=int(cfg_data.get("min_school_count", 10)),
        ridge_alpha=float(cfg_data.get("ridge_alpha", 1.0)),
        bootstrap_iterations=int(args.bootstrap_iterations if args.bootstrap_iterations is not None else cfg_data.get("bootstrap_iterations", 300)),
        random_seed=int(cfg_data.get("random_seed", 42)),
    )
    result_df, meta = analyze_value_added(admissions_df, maturita_df, cfg)
    meta["source_provenance"] = run_meta

    analysis_csv = output_dir / "school_value_added.csv"
    analysis_meta_json = output_dir / "school_value_added.methodology.json"
    report_base = args.report_base_name
    result_df.to_csv(analysis_csv, index=False)
    dump_json(analysis_meta_json, meta)
    write_report(result_df, meta, output_dir=output_dir, base_name=report_base)

    run_manifest = {
        "admissions_normalized": str(admissions_csv),
        "maturita_normalized": str(maturita_csv),
        "analysis_csv": str(analysis_csv),
        "analysis_methodology": str(analysis_meta_json),
        "report_html": str(Path(output_dir) / f"{report_base}.html"),
        "report_csv": str(Path(output_dir) / f"{report_base}.csv"),
        "report_methodology": str(Path(output_dir) / f"{report_base}.methodology.json"),
    }
    dump_json(Path(output_dir) / "run_outputs.json", run_manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gymva", description="School-level 8-year gymnasium adjusted association analysis (JPZ vs maturita)")
    p.add_argument("--verbose", action="store_true")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_down = sp.add_parser("download", help="Download source files from a static manifest")
    p_down.add_argument("--manifest", default="config/sources.json")
    p_down.add_argument("--raw-dir", default="data/raw")
    p_down.add_argument("--meta-out", default="data/raw/download_metadata.json")
    p_down.add_argument("--timeout", type=int, default=60)
    p_down.add_argument("--retries", type=int, default=3)
    p_down.add_argument("--require-excel", action="store_true")
    p_down.set_defaults(func=cmd_download)

    p_disc = sp.add_parser("discover-columns", help="Inspect source columns and sample rows")
    p_disc.add_argument("--input", required=True)
    p_disc.add_argument("--sheet")
    p_disc.set_defaults(func=cmd_discover_columns)

    p_ing = sp.add_parser("ingest", help="Normalize admissions and maturita tables")
    p_ing.add_argument("--manifest", default="config/sources.json")
    p_ing.add_argument("--raw-dir", default="data/raw")
    p_ing.add_argument("--mapping", default="config/column_mappings.json")
    p_ing.add_argument("--eight-year-filter", default="config/eight_year_filter.json")
    p_ing.add_argument("--admissions-input")
    p_ing.add_argument("--maturita-input")
    p_ing.add_argument("--admissions-source-id")
    p_ing.add_argument("--maturita-source-id")
    p_ing.add_argument("--admissions-sheet")
    p_ing.add_argument("--maturita-sheet")
    p_ing.add_argument("--out-dir", default="data/normalized")
    p_ing.set_defaults(func=cmd_ingest)

    p_an = sp.add_parser("analyze", help="Fit adjusted school-level residual model")
    p_an.add_argument("--admissions", default="data/normalized/admissions_normalized.csv")
    p_an.add_argument("--maturita", default="data/normalized/maturita_normalized.csv")
    p_an.add_argument("--analysis-config", default="config/analysis.json")
    p_an.add_argument("--output", default="data/analysis/school_value_added.csv")
    p_an.add_argument("--meta-output", default="data/analysis/school_value_added.methodology.json")
    p_an.set_defaults(func=cmd_analyze)

    p_rep = sp.add_parser("report", help="Generate CSV/JSON/HTML report")
    p_rep.add_argument("--analysis-csv", default="data/analysis/school_value_added.csv")
    p_rep.add_argument("--methodology-json", default="data/analysis/school_value_added.methodology.json")
    p_rep.add_argument("--output-dir", default="data/report")
    p_rep.add_argument("--base-name", default="gymva_report")
    p_rep.set_defaults(func=cmd_report)

    p_archive = sp.add_parser("archive", help="Download and freeze official CERMAT archive (network only here)")
    p_archive.add_argument("--archive-root", default="data/archive")
    p_archive.add_argument("--freeze-id", default=None)
    p_archive.add_argument("--year-start", type=int, default=2016)
    p_archive.add_argument("--year-end", type=int, default=2026)
    p_archive.add_argument("--refresh", action="store_true")
    p_archive.add_argument("--timeout", type=int, default=60)
    p_archive.add_argument("--retries", type=int, default=3)
    p_archive.add_argument("--source-override", default=None, help="JSON map source_id/url -> local file path")
    p_archive.set_defaults(func=cmd_archive)

    p_archive_official = sp.add_parser("archive-official", help="Alias for archive")
    p_archive_official.add_argument("--archive-root", default="data/archive")
    p_archive_official.add_argument("--freeze-id", default=None)
    p_archive_official.add_argument("--year-start", type=int, default=2016)
    p_archive_official.add_argument("--year-end", type=int, default=2026)
    p_archive_official.add_argument("--refresh", action="store_true")
    p_archive_official.add_argument("--timeout", type=int, default=60)
    p_archive_official.add_argument("--retries", type=int, default=3)
    p_archive_official.add_argument("--source-override", default=None, help="JSON map source_id/url -> local file path")
    p_archive_official.set_defaults(func=cmd_archive)

    p_an_local = sp.add_parser("analyze-local", help="Analyze frozen archive strictly locally")
    p_an_local.add_argument("--archive", required=True)
    p_an_local.set_defaults(func=cmd_analyze_local)

    p_rep_local = sp.add_parser("report-local", help="Render the local archive dashboard from frozen artifacts")
    p_rep_local.add_argument("--archive", required=True)
    p_rep_local.set_defaults(func=cmd_report_local)

    p_run = sp.add_parser("run", help="Supported path: archive + analyze-local (explicit, no hidden fetch)")
    p_run.add_argument("--archive", default=None)
    p_run.add_argument("--archive-root", default="data/archive")
    p_run.add_argument("--freeze-id", default=None)
    p_run.add_argument("--year-start", type=int, default=2016)
    p_run.add_argument("--year-end", type=int, default=2026)
    p_run.add_argument("--allow-network-fetch", action="store_true")
    p_run.add_argument("--source-override", default=None)
    p_run.add_argument("--refresh", action="store_true")
    p_run.add_argument("--retries", type=int, default=3)
    p_run.add_argument("--timeout", type=int, default=60)
    p_run.set_defaults(func=cmd_run)

    p_run_legacy = sp.add_parser("run-legacy", help="Legacy autonomous pipeline")
    p_run_legacy.add_argument("--data-dir", default="data")
    p_run_legacy.add_argument("--output-dir", default="output")
    p_run_legacy.add_argument("--mapping", default="config/column_mappings.json")
    p_run_legacy.add_argument("--eight-year-filter", default="config/eight_year_filter.json")
    p_run_legacy.add_argument("--analysis-config", default="config/analysis.json")
    p_run_legacy.add_argument("--entry-years", default=None, help="Comma-separated JPZ entry years (optional)")
    p_run_legacy.add_argument("--graduation-years", default=None, help="Comma-separated maturita graduation years (optional)")
    p_run_legacy.add_argument("--cohort-lag", type=int, default=8)
    refresh_group = p_run_legacy.add_mutually_exclusive_group()
    refresh_group.add_argument("--refresh", dest="refresh", action="store_true", help="Force re-download even if files already exist")
    refresh_group.add_argument("--no-refresh", dest="refresh", action="store_false", help="Reuse existing downloaded files when available")
    p_run_legacy.set_defaults(refresh=False)
    p_run_legacy.add_argument("--retries", type=int, default=3)
    p_run_legacy.add_argument("--timeout", type=int, default=60)
    p_run_legacy.add_argument("--bootstrap-iterations", type=int, default=None)
    p_run_legacy.add_argument("--use-all-pairs", action="store_true", help="Use all valid cohort pairs instead of latest pair")
    p_run_legacy.add_argument("--report-base-name", default="gymva_report")
    p_run_legacy.add_argument("--jpz-landing-url", default=JPZ_LANDING_URL)
    p_run_legacy.add_argument("--maturita-landing-url", default=MZ_LANDING_URL)
    p_run_legacy.add_argument("--discovery-json", default=None, help="Offline discovery override JSON for testing/reproducibility")
    p_run_legacy.add_argument("--maturita-sheet", default=None)
    p_run_legacy.set_defaults(func=cmd_run_legacy)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    main()
