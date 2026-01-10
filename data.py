"""
Pier 26 (USGS-01376520) — Salinity dashboard (from 2019 onward)
- Pull continuous salinity (parameter 90860, statistic 00011) in chunks
- Aggregate to daily mean
- Save CSV + interactive HTML dashboard
- Dashboard has Start/End date inputs with min/max constrained to actual data availability

Install:
  /usr/local/bin/python3 -m pip install requests pandas plotly

Run:
  /usr/local/bin/python3 "/Users/samialemfadli/Desktop/HRECSO Data/data.py"

Outputs:
  pier26_salinity_daily_from2019.csv
  pier26_salinity_dashboard_from2019.html
"""

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import plotly.graph_objects as go


# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pier26-salinity-dashboard/1.0"})

PIER26_MONITORING_LOCATION_ID = "USGS-01376520"
SALINITY_PCODE = "90860"
CONT_STAT_ID = "00011"

# You asked to "go back to 2019" for salinity data:
REQUESTED_START = date(2019, 1, 1)  # we will still constrain the GUI to actual available data
CHUNK_DAYS = 180


# ----------------------------
# HTTP helpers
# ----------------------------
def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _date_range_iso(start: date, end: date) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


# ----------------------------
# Fetch continuous (single request, with pagination)
# ----------------------------
def fetch_continuous_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = CONT_STAT_ID,
) -> pd.DataFrame:
    url = f"{OGC_BASE}/collections/continuous/items"
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": parameter_code,
        "statistic_id": statistic_id,
        "time": _date_range_iso(start, end),
        "f": "json",
        "limit": 10000,
    }

    rows: List[Tuple[pd.Timestamp, float]] = []
    next_url: Optional[str] = url
    next_params: Optional[Dict[str, Any]] = params

    while next_url:
        out = _get_json(next_url, params=next_params)

        for feat in out.get("features", []) or []:
            p = feat.get("properties", {}) or {}
            t = p.get("time")
            v = p.get("value")
            if t is None or v is None:
                continue
            rows.append((pd.to_datetime(t, utc=True), float(v)))

        next_link = None
        for link in out.get("links", []) or []:
            if link.get("rel") == "next" and link.get("href"):
                next_link = link["href"]
                break

        if next_link:
            next_url = next_link
            next_params = None  # next link already includes query
        else:
            next_url = None

    return pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")


# ----------------------------
# Fetch continuous (chunked, avoids long-range 400s)
# ----------------------------
def fetch_continuous_values_chunked(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = CONT_STAT_ID,
    chunk_days: int = 180,
) -> pd.DataFrame:
    all_parts: List[pd.DataFrame] = []
    cur = start

    while cur <= end:
        cur_end = min(end, cur + timedelta(days=chunk_days))

        df_part = fetch_continuous_values(
            monitoring_location_id=monitoring_location_id,
            parameter_code=parameter_code,
            start=cur,
            end=cur_end,
            statistic_id=statistic_id,
        )
        all_parts.append(df_part)

        print(f"Fetched {cur} to {cur_end}: {len(df_part)} rows")
        cur = cur_end + timedelta(days=1)

    # Keep only non-empty parts (prevents concat warnings & speeds up)
    non_empty = [p for p in all_parts if not p.empty]
    if not non_empty:
        return pd.DataFrame(columns=["time", "value"])

    df = pd.concat(non_empty, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------------
# Aggregation
# ----------------------------
def to_daily_mean(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index("time")["value"].resample("D").mean()


# ----------------------------
# Dashboard with date inputs constrained to available data
# ----------------------------
def build_salinity_dashboard_html(
    daily_df: pd.DataFrame,
    title: str,
    output_html: str,
) -> str:
    if daily_df.empty:
        raise ValueError("No salinity data available for the requested window.")

    # Determine actual available range (THIS is what we enforce in the GUI)
    data_start = daily_df.index.min().date().isoformat()
    data_end = daily_df.index.max().date().isoformat()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_df.index,
            y=daily_df["salinity"],
            mode="lines",
            name="Salinity",
        )
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=650,
        margin=dict(l=60, r=30, t=80, b=60),
    )
    fig.update_xaxes(
        title_text="Date",
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    fig.update_yaxes(title_text="Salinity (sensor units)")

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="salinityPlot")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 18px; }}
    .controls {{ display: flex; gap: 10px; align-items: end; flex-wrap: wrap; margin-bottom: 10px; }}
    .control {{ display: flex; flex-direction: column; gap: 6px; }}
    input[type="date"] {{ padding: 8px; font-size: 14px; }}
    button {{ padding: 9px 12px; font-size: 14px; cursor: pointer; }}
    .hint {{ color: #666; font-size: 12px; margin-top: 6px; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 14px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="controls">
      <div class="control">
        <label for="startDate"><b>Start date</b></label>
        <input id="startDate" type="date"
               value="{data_start}"
               min="{data_start}"
               max="{data_end}" />
      </div>

      <div class="control">
        <label for="endDate"><b>End date</b></label>
        <input id="endDate" type="date"
               value="{data_end}"
               min="{data_start}"
               max="{data_end}" />
      </div>

      <button id="applyBtn">Apply</button>
      <button id="resetBtn">Reset (All)</button>
    </div>

    <div class="hint">
      Available data range: <b>{data_start}</b> to <b>{data_end}</b>.
      Use the date fields or the range slider under the chart.
    </div>
  </div>

  <div style="height: 14px;"></div>

  {plot_div}

  <script>
    const plotId = "salinityPlot";
    const startInput = document.getElementById("startDate");
    const endInput = document.getElementById("endDate");
    const applyBtn = document.getElementById("applyBtn");
    const resetBtn = document.getElementById("resetBtn");

    const dataStart = "{data_start}";
    const dataEnd = "{data_end}";

    function clampDate(d) {{
      if (d < dataStart) return dataStart;
      if (d > dataEnd) return dataEnd;
      return d;
    }}

    function applyRange(start, end) {{
      start = clampDate(start);
      end = clampDate(end);

      if (start > end) {{
        alert("Start date must be ≤ End date.");
        return;
      }}

      // Keep inputs consistent with clamped values
      startInput.value = start;
      endInput.value = end;

      Plotly.relayout(plotId, {{
        "xaxis.range": [start, end]
      }});
    }}

    applyBtn.addEventListener("click", () => {{
      applyRange(startInput.value, endInput.value);
    }});

    resetBtn.addEventListener("click", () => {{
      startInput.value = dataStart;
      endInput.value = dataEnd;
      applyRange(dataStart, dataEnd);
    }});
  </script>
</body>
</html>
"""
    out_path = Path(output_html)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # Pull from 2019 onward (user request)
    start = REQUESTED_START
    end = date.today()

    print(f"Requested: {start} to {end}")
    print(f"Using:     {start} to {end}")

    # Fetch continuous in chunks
    sal_cont = fetch_continuous_values_chunked(
        PIER26_MONITORING_LOCATION_ID,
        SALINITY_PCODE,
        start,
        end,
        statistic_id=CONT_STAT_ID,
        chunk_days=CHUNK_DAYS,
    )

    if sal_cont.empty:
        raise RuntimeError("No salinity observations returned from 2019 onward for this site/parameter.")

    # Aggregate to daily mean
    sal_daily = to_daily_mean(sal_cont)
    daily = pd.DataFrame({"salinity": sal_daily})
    daily.index.name = "date_utc"

    # Save CSV
    csv_path = BASE_DIR / "pier26_salinity_daily_from2019.csv"
    daily.to_csv(csv_path)

    # Build dashboard (date inputs constrained to actual data min/max)
    html_path = BASE_DIR / "pier26_salinity_dashboard_from2019.html"
    build_salinity_dashboard_html(
        daily_df=daily,
        title="Pier 26 (USGS-01376520): Salinity — from 2019 onward",
        output_html=str(html_path),
    )

    first = daily.index.min().date().isoformat()
    last = daily.index.max().date().isoformat()
    print(f"Actual available salinity range: {first} to {last}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved dashboard: {html_path}")
    print(f"Open in browser: open \"{html_path}\"")


if __name__ == "__main__":
    main()

    print("Done!")

