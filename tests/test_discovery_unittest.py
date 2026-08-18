from __future__ import annotations

import unittest
from unittest.mock import patch

from gymnazium_value_added.discovery import discover_jpz_sources, discover_maturita_sources, select_cohort_pairs


class DiscoveryTests(unittest.TestCase):
    def test_discovery_parses_direct_cermat_links_only(self) -> None:
        jpz_html = """
        <html><body>
          <a href="/files/files/JPZ/agregovana_data_skoly/PZ2026_kolo1_skolobory_prihlasky.xlsx">P</a>
          <a href="https://data.cermat.cz/files/files/JPZ/agregovana_data_skoly/PZ2026_kolo1_skolobory_kapacity.xlsx">K</a>
          <a href="https://data.cermat.cz/files/files/JPZ/agregovana_data_skoly/JPZ2017_skoly-skolobory_vysledky.xlsx">H</a>
          <a href="https://example.com/PZ2026_kolo1_skolobory_vysledky.xlsx">external</a>
        </body></html>
        """
        mz_html = """
        <html><body>
          <a href="/files/files/MZ/agregovana_data_skoly/MZ2034j_SC_skolobory.xlsx">MZ</a>
        </body></html>
        """

        with patch("gymnazium_value_added.discovery._fetch_html", side_effect=[jpz_html, mz_html]):
            jpz = discover_jpz_sources("https://data.cermat.cz/menu/jpz")
            mz = discover_maturita_sources("https://data.cermat.cz/menu/mz")

        self.assertEqual(len(jpz), 3)
        self.assertEqual({x.kind for x in jpz}, {"prihlasky", "kapacity", "vysledky"})
        self.assertEqual({x.year for x in jpz}, {2017, 2026})
        self.assertEqual(len(mz), 1)
        self.assertEqual(mz[0].year, 2034)

    def test_pair_selection_respects_lag(self) -> None:
        jpz_html = """
        <html><body>
          <a href="/files/files/JPZ/agregovana_data_skoly/PZ2018_kolo1_skolobory_prihlasky.xlsx">P</a>
          <a href="/files/files/JPZ/agregovana_data_skoly/PZ2018_kolo1_skolobory_kapacity.xlsx">K</a>
        </body></html>
        """
        mz_html = """
        <html><body>
          <a href="/files/files/MZ/agregovana_data_skoly/MZ2026j_SC_skolobory.xlsx">MZ</a>
        </body></html>
        """
        with patch("gymnazium_value_added.discovery._fetch_html", side_effect=[jpz_html, mz_html]):
            jpz = discover_jpz_sources("https://data.cermat.cz/menu/jpz")
            mz = discover_maturita_sources("https://data.cermat.cz/menu/mz")
        pairs = select_cohort_pairs(jpz, mz, cohort_lag_years=8)
        self.assertEqual(pairs, [{"entry_year": 2018, "graduation_year": 2026, "jpz_mode": "triplet"}])


if __name__ == "__main__":
    unittest.main()
