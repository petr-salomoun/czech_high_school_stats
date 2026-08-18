from __future__ import annotations

import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from gymnazium_value_added.io import DownloadError, download_first_valid_excel, download_url


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/bad404.xlsx"):
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html>Not found</html>")
            return
        if self.path.endswith("/html.xlsx"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html>Still HTML</html>")
            return
        if self.path.endswith("/ok.xlsx"):
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.end_headers()
            # Zip signature enough for validator; parser-level checks are elsewhere.
            self.wfile.write(b"PK\x03\x04fixture-xlsx")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class IoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_rejects_html_when_excel_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.xlsx"
            with self.assertRaises(RuntimeError):
                download_url(f"{self.base}/html.xlsx", target, require_excel=True, retries=1)

    def test_reports_all_attempts_for_first_valid_excel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.xlsx"
            with self.assertRaises(DownloadError) as ctx:
                download_first_valid_excel([f"{self.base}/bad404.xlsx", f"{self.base}/html.xlsx"], target, retries=1)
            self.assertEqual(len(ctx.exception.attempts), 2)

    def test_download_first_valid_excel_accepts_real_excel_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ok.xlsx"
            used, meta = download_first_valid_excel(
                [f"{self.base}/bad404.xlsx", f"{self.base}/ok.xlsx"],
                target,
                retries=1,
            )
            self.assertTrue(target.exists())
            self.assertTrue(str(used).endswith("/ok.xlsx"))
            self.assertEqual(meta["filename"], "ok.xlsx")


if __name__ == "__main__":
    unittest.main()
