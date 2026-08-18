from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib import request
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError


LOGGER = logging.getLogger(__name__)

EXCEL_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/octet-stream",
    "binary/octet-stream",
}


class DownloadError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, str | int]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _looks_like_html(blob: bytes) -> bool:
    sample = blob[:512].strip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html")


def _looks_like_excel(blob: bytes) -> bool:
    if len(blob) < 8:
        return False
    # XLSX (ZIP): PK\x03\x04 ; legacy XLS OLE compound: D0 CF 11 E0 A1 B1 1A E1
    return blob.startswith(b"PK\x03\x04") or blob.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))


def _file_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name.strip()
    return name or "downloaded_file"


def _download_once(url: str, timeout: int = 60) -> tuple[bytes, dict[str, str], int]:
    req = request.Request(url, headers={"User-Agent": "gymva/0.2"})
    with request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        headers = {k: v for k, v in resp.headers.items()}
        raw_status = getattr(resp, "status", 200)
        status = int(raw_status) if raw_status is not None else 200
    return data, headers, status


def download_url(
    url: str,
    target: Path,
    timeout: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    require_excel: bool = False,
) -> dict[str, str | int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s -> %s", url, target)

    last_exc: Exception | None = None
    data: bytes | None = None
    headers: dict[str, str] = {}
    status = 0
    for attempt in range(1, retries + 1):
        try:
            data, headers, status = _download_once(url, timeout=timeout)
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                raise RuntimeError(f"Download failed for {url} after {retries} attempts: {exc}") from exc
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning("Download attempt %d/%d failed for %s (%s), retrying in %.1fs", attempt, retries, url, exc, sleep_for)
            time.sleep(sleep_for)

    if data is None:
        raise RuntimeError(f"Download failed for {url}: {last_exc}")

    content_type = str(headers.get("Content-Type", "")).split(";")[0].strip().lower()
    if require_excel:
        if _looks_like_html(data):
            raise RuntimeError(f"Downloaded HTML instead of Excel file: {url}")
        if content_type and content_type not in EXCEL_CONTENT_TYPES and "excel" not in content_type and "spreadsheetml" not in content_type:
            raise RuntimeError(f"Unexpected content type for Excel file: url={url} content_type={content_type}")
        if not _looks_like_excel(data):
            raise RuntimeError(f"Downloaded content is not a valid Excel signature: {url}")

    with NamedTemporaryFile("wb", delete=False, dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)

    digest = sha256_of_file(target)
    return {
        "status": int(status),
        "bytes": target.stat().st_size,
        "sha256": digest,
        "etag": headers.get("ETag", ""),
        "last_modified": headers.get("Last-Modified", ""),
        "content_type": headers.get("Content-Type", ""),
        "filename": _file_name_from_url(url),
    }


def download_first_valid_excel(
    urls: list[str],
    target: Path,
    timeout: int = 60,
    retries: int = 3,
) -> tuple[str, dict[str, str | int]]:
    attempts: list[dict[str, str | int]] = []
    for url in urls:
        try:
            meta = download_url(url, target, timeout=timeout, retries=retries, require_excel=True)
            attempts.append({"url": url, "status": "ok", "http_status": int(meta.get("status", 0))})
            return url, {**meta, "attempts": attempts}
        except Exception as exc:
            attempts.append({"url": url, "status": "failed", "error": str(exc)})
            continue
    raise DownloadError("All candidate URLs failed to download as Excel.", attempts)
