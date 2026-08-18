from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def write_report(
    result_df: pd.DataFrame,
    methodology: dict[str, Any],
    output_dir: str | Path,
    base_name: str = "gymva_report",
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{base_name}.csv"
    json_path = out / f"{base_name}.methodology.json"
    html_path = out / f"{base_name}.html"

    result_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for i, row in result_df.reset_index(drop=True).iterrows():
        rows.append(
            "<tr>"
            f"<td>{i+1}</td>"
            f"<td>{html.escape(str(row['school_name']))}</td>"
            f"<td>{html.escape(_text(row.get('address'), row.get('address_raw')))}</td>"
            f"<td>{html.escape(str(row['school_id']))}</td>"
            f"<td>{_fmt(row['mean_selection_metric'])}</td>"
            f"<td>{_fmt(row['mean_admission_score'])}</td>"
            f"<td>{_fmt(row['observed_outcome'])}</td>"
            f"<td>{_fmt(row['expected_outcome'])}</td>"
            f"<td>{_fmt(row['value_added'])}</td>"
            f"<td>{_fmt(row['value_added_ci_low'])} to {_fmt(row['value_added_ci_high'])}</td>"
            f"<td>{html.escape(str(row['quality_flag']))}</td>"
            "</tr>"
        )

    outcome_label = str(methodology.get("outcome_definition", "unknown"))
    outcome_note = (
        "Outcome metric: mean_score (preferred)."
        if outcome_label == "mean_score"
        else "Outcome metric: pass_rate (mean_score not available in source schema)."
    )

    html_text = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>GymVA report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px; font-size: 14px; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>8-year Gymnasium Comparison: Selectivity vs Adjusted Association</h1>
  <p><strong>Important:</strong> This is a repeated cross-sectional school-level adjusted association model, not an individual-level causal estimate.</p>
  <p>The model adjusts maturita outcomes using entry selection metrics when observed (applications/capacity), supports historical JPZ score-only cohorts with missing selectivity flags, and controls for subject and graduation year.</p>
  <p>{html.escape(str(methodology.get("methodology", "")))}</p>
  <p>{html.escape(outcome_note)}</p>

  <h2>School ranking by adjusted residual value</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>School</th><th>Address</th><th>ID</th><th>Selection metric</th><th>Avg entry score</th>
        <th>Observed outcome</th><th>Expected outcome</th><th>Residual value</th><th>95% CI</th><th>Data quality</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <h2>Methodological limitations</h2>
  <ul>
    <li>Without applicant-to-graduate individual linkage, this is not causal inference.</li>
    <li>Results depend on correct column mapping and CERMAT data completeness.</li>
    <li>Schools with low candidate counts are marked in quality_flag.</li>
  </ul>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "html": str(html_path),
    }
