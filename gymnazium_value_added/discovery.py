from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib import request
from urllib.parse import urljoin, urlparse


JPZ_LANDING_URL = "https://data.cermat.cz/menu/data-a-analyticke-vystupy-jednotna-prijimaci-zkouska/agregovana-data-jpz"
MZ_LANDING_URL = "https://data.cermat.cz/menu/maturitni-zkouska/agregovana-data"


@dataclass(frozen=True)
class SourceCandidate:
    dataset: str
    year: int | None
    kind: str
    url: str


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _fetch_html(url: str, timeout: int = 60) -> str:
    req = request.Request(url, headers={"User-Agent": "gymva/0.2"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return body


def _extract_links(html_text: str, base_url: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(html_text)
    normalized: list[str] = []
    for href in parser.hrefs:
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized.append(full)
    return list(dict.fromkeys(normalized))


def _is_cermat_data_file(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "data.cermat.cz":
        return False
    return parsed.path.lower().endswith((".xlsx", ".xls")) and "/files/" in parsed.path.lower()


def discover_jpz_sources(landing_url: str = JPZ_LANDING_URL, timeout: int = 60) -> list[SourceCandidate]:
    html_text = _fetch_html(landing_url, timeout=timeout)
    links = _extract_links(html_text, landing_url)
    out: list[SourceCandidate] = []
    patterns = [
        re.compile(r"/(?:PZ|JPZ)(?P<year>\d{4})_kolo1_skolobory_(?P<kind>prihlasky|kapacity|vysledky)\.xlsx$", re.IGNORECASE),
        re.compile(r"/(?:PZ|JPZ)(?P<year>\d{4})_skoly-skolobory_vysledky\.xlsx$", re.IGNORECASE),
    ]
    for link in links:
        if not _is_cermat_data_file(link):
            continue
        path = urlparse(link).path
        for pattern in patterns:
            m = pattern.search(path)
            if not m:
                continue
            year = int(m.group("year"))
            kind = m.groupdict().get("kind", "vysledky")
            out.append(SourceCandidate(dataset="jpz", year=year, kind=kind.lower(), url=link))
            break
    return sorted(out, key=lambda x: (x.year, x.kind, x.url))


def discover_maturita_sources(landing_url: str = MZ_LANDING_URL, timeout: int = 60) -> list[SourceCandidate]:
    html_text = _fetch_html(landing_url, timeout=timeout)
    links = _extract_links(html_text, landing_url)
    out: list[SourceCandidate] = []
    pattern = re.compile(r"/MZ(?P<year>\d{4})(?:[^/]*)_skolobory\.xlsx$", re.IGNORECASE)
    for link in links:
        if not _is_cermat_data_file(link):
            continue
        m = pattern.search(urlparse(link).path)
        if not m:
            continue
        year = int(m.group("year"))
        out.append(SourceCandidate(dataset="maturita", year=year, kind="aggregate", url=link))
    return sorted(out, key=lambda x: (x.year, x.url))


def discover_all_sources(
    jpz_landing_url: str = JPZ_LANDING_URL,
    maturita_landing_url: str = MZ_LANDING_URL,
    timeout: int = 60,
) -> dict[str, list[SourceCandidate]]:
    return {
        "jpz": discover_jpz_sources(jpz_landing_url, timeout=timeout),
        "maturita": discover_maturita_sources(maturita_landing_url, timeout=timeout),
    }


def select_cohort_pairs(
    jpz_sources: list[SourceCandidate],
    maturita_sources: list[SourceCandidate],
    cohort_lag_years: int,
    entry_years: list[int] | None = None,
    graduation_years: list[int] | None = None,
) -> list[dict[str, int]]:
    jpz_by_year: dict[int, set[str]] = {}
    for s in jpz_sources:
        if s.year is None:
            continue
        jpz_by_year.setdefault(s.year, set()).add(s.kind)

    valid_jpz_modes: dict[int, str] = {}
    for year, kinds in jpz_by_year.items():
        if {"prihlasky", "kapacity"}.issubset(kinds):
            valid_jpz_modes[year] = "triplet"
        elif "vysledky" in kinds:
            valid_jpz_modes[year] = "results_only"
    maturita_years = {s.year for s in maturita_sources if s.year is not None}

    pairs: list[dict[str, int]] = []
    for grad_year in sorted(maturita_years):
        entry_year = grad_year - cohort_lag_years
        mode = valid_jpz_modes.get(entry_year)
        if mode is None:
            continue
        if entry_years is not None and entry_year not in set(entry_years):
            continue
        if graduation_years is not None and grad_year not in set(graduation_years):
            continue
        pairs.append({"entry_year": entry_year, "graduation_year": grad_year, "jpz_mode": mode})
    return pairs
