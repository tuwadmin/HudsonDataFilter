"""
Pier 26 (USGS-01376520) — Last 5 years dashboard for Salinity + Precipitation (USGS OGC API)

Features:
- Pull continuous salinity (90860) and precipitation (00045) from USGS OGC API
- Chunked requests to avoid 400 errors for long time windows
- Aggregate daily:
    Salinity = daily mean
    Precip   = daily total (sum)
- Save CSV with both series
- Build interactive HTML dashboard (index.html) with:
    - Start/End date inputs (bounded to available data window)
    - k·σ inputs for each series
    - Data coverage % for each series in the displayed interval
    - Toggle: precip highlight only above average
    - Rolling 7-day mean overlays for both series
    - Highlights values outside mean ± k·σ (or above mean + k·σ for precip)

Install:
  /usr/local/bin/python3 -m pip install requests pandas plotly

Run:
  /usr/local/bin/python3 "/Users/samialemfadli/Desktop/HRECSO Data/data.py"

Outputs (in same folder as this script):
  pier26_last5y_salinity_precip_daily.csv
  index.html   (ready for GitHub Pages)
"""

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple

import json
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pier26-dashboard/1.0"})

PIER26_MONITORING_LOCATION_ID = "USGS-01376520"

# Parameter codes
SALINITY_PCODE = "90860"
PRECIP_PCODE = "00045"

# Statistic ID that worked in your earlier runs
CONT_STAT_ID = "00011"

# Chunking prevents long-range 400 responses
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
# Continuous fetch (single window, paginated)
# ----------------------------
def fetch_continuous_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = CONT_STAT_ID,
) -> pd.DataFrame:
    """
    Fetch continuous values for one time window, following 'next' pagination links.
    Returns DataFrame columns: time (UTC), value (float)
    """
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
            next_params = None  # already encoded in next link
        else:
            next_url = None

    return pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")


def fetch_continuous_values_chunked(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = CONT_STAT_ID,
    chunk_days: int = 180,
) -> pd.DataFrame:
    """
    Fetch continuous values in multiple smaller time chunks.
    This avoids 400 errors for long windows and keeps requests manageable.
    """
    parts: List[pd.DataFrame] = []
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

        print(f"Fetched {parameter_code} {cur} to {cur_end}: {len(df_part)} rows")
        if not df_part.empty:
            parts.append(df_part)

        cur = cur_end + timedelta(days=1)

    if not parts:
        return pd.DataFrame(columns=["time", "value"])

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------------
# Aggregation
# ----------------------------
def to_daily_mean(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index("time")["value"].resample("D").mean()


def to_daily_total(df: pd.DataFrame) -> pd.Series:
    """
    For precipitation, daily total is usually most useful for event analysis.
    Summing within day behaves well whether values are sub-daily or already daily.
    """
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index("time")["value"].resample("D").sum()


# ----------------------------
# Dashboard builder (HTML + JS)
# ----------------------------
def build_dashboard_html(
    daily: pd.DataFrame,
    output_html: str,
    title: str = "Pier 26 (USGS-01376520): Salinity + Precipitation — last 5 years",
) -> str:
    if daily.empty:
        raise ValueError("No daily data to plot.")

    data_start = daily.index.min().date().isoformat()
    data_end = daily.index.max().date().isoformat()

    dates = [d.strftime("%Y-%m-%d") for d in daily.index]
    sal = [None if pd.isna(v) else float(v) for v in daily["salinity"].tolist()]
    pr = [None if pd.isna(v) else float(v) for v in daily["precip_in"].tolist()]

    # Plotly figure skeleton; JS fills traces based on user inputs
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Salinity (daily mean)", "Precipitation (daily total)"),
    )

    # Trace order (JS uses these indices):
    # 0 sal line
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="Salinity", opacity=0.85),
        row=1, col=1
    )

    # 1 sal highlight markers (outliers)
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="markers", name="Salinity (outliers)",
            marker=dict(size=7, symbol="circle-open"),
            opacity=0.9
        ),
        row=1, col=1
    )

    # 2 sal rolling 7d mean
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="Salinity (7d mean)", opacity=0.9),
        row=1, col=1
    )

    # 3 precip bars (base)
    fig.add_trace(
        go.Bar(x=[], y=[], name="Precip (in)", opacity=0.45),
        row=2, col=1
    )

    # 4 precip highlight bars (outliers)
    fig.add_trace(
        go.Bar(x=[], y=[], name="Precip (outliers)", opacity=0.9),
        row=2, col=1
    )

    # 5 precip rolling 7d mean
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="Precip (7d mean)", opacity=0.9),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=950,
        margin=dict(l=60, r=30, t=90, b=60),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Salinity (sensor units)", row=1, col=1)
    fig.update_yaxes(title_text="Inches", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1, rangeslider=dict(visible=True))

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot")
    data_json = json.dumps({"dates": dates, "salinity": sal, "precip": pr})

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 18px; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 14px; }}
    .controls {{ display: flex; gap: 12px; align-items: end; flex-wrap: wrap; }}
    .control {{ display: flex; flex-direction: column; gap: 6px; }}
    input[type="date"], input[type="number"] {{ padding: 8px; font-size: 14px; min-width: 160px; }}
    button {{ padding: 9px 12px; font-size: 14px; cursor: pointer; }}
    .hint {{ color: #666; font-size: 12px; margin-top: 8px; line-height: 1.4; }}
    .metrics {{ display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 10px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .metricline {{ font-size: 13px; color: #111; }}
    .metricline b {{ font-weight: 600; }}
    .checkline {{ display: flex; gap: 8px; align-items: center; font-size: 13px; color: #111; padding-bottom: 2px; }}
    @media (max-width: 900px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

  <div class="card">
    <div class="controls">
      <div class="control">
        <label for="startDate"><b>Start date</b></label>
        <input id="startDate" type="date" value="{data_start}" min="{data_start}" max="{data_end}" />
      </div>

      <div class="control">
        <label for="endDate"><b>End date</b></label>
        <input id="endDate" type="date" value="{data_end}" min="{data_start}" max="{data_end}" />
      </div>

      <div class="control">
        <label for="kSal"><b>Salinity highlight (k·σ)</b></label>
        <input id="kSal" type="number" step="0.5" value="1.0" min="0" />
      </div>

      <div class="control">
        <label for="kPr"><b>Precip highlight (k·σ)</b></label>
        <input id="kPr" type="number" step="0.5" value="1.0" min="0" />
      </div>

      <div class="control">
        <div class="checkline">
          <input id="prAboveOnly" type="checkbox" checked />
          <label for="prAboveOnly"><b>Precip: highlight only above avg</b></label>
        </div>
        <div class="hint" style="margin-top:0;">(If off: highlights both below and above avg)</div>
      </div>

      <button id="applyBtn">Apply</button>
      <button id="resetBtn">Reset (All)</button>
    </div>

    <div class="metrics grid2">
      <div class="metricline" id="salStats">Salinity avg: — | std: — | coverage: — | highlight: —</div>
      <div class="metricline" id="prStats">Precip avg: — | std: — | coverage: — | highlight: —</div>
    </div>

    <div class="hint">
      Highlights use <b>mean ± (k·σ)</b> over the displayed interval.
      Rolling overlays are <b>7-day means</b> (computed over available values).
      Available data window: {data_start} to {data_end}.
    </div>
  </div>

  <div style="height: 14px;"></div>

  {plot_div}

<script>
  const DATA = {data_json};

  const startInput = document.getElementById("startDate");
  const endInput   = document.getElementById("endDate");
  const kSalInput  = document.getElementById("kSal");
  const kPrInput   = document.getElementById("kPr");
  const prAboveOnlyInput = document.getElementById("prAboveOnly");
  const applyBtn   = document.getElementById("applyBtn");
  const resetBtn   = document.getElementById("resetBtn");
  const salStatsEl = document.getElementById("salStats");
  const prStatsEl  = document.getElementById("prStats");

  const plotId = "plot";
  const dataStart = "{data_start}";
  const dataEnd   = "{data_end}";

  function clampDate(d) {{
    if (d < dataStart) return dataStart;
    if (d > dataEnd) return dataEnd;
    return d;
  }}

  function meanStd(values) {{
    const n = values.length;
    if (n === 0) return {{mean: NaN, std: NaN}};
    let sum = 0;
    for (const v of values) sum += v;
    const mean = sum / n;
    let ss = 0;
    for (const v of values) {{
      const diff = v - mean;
      ss += diff * diff;
    }}
    const std = Math.sqrt(ss / n); // population std
    return {{mean, std}};
  }}

  function fmt(x) {{
    if (!Number.isFinite(x)) return "—";
    if (Math.abs(x) < 0.001 && x !== 0) return x.toExponential(3);
    return x.toFixed(3);
  }}

  function pct(numer, denom) {{
    if (denom <= 0) return "—";
    return (100 * numer / denom).toFixed(1) + "%";
  }}

  function rollingMean(values, windowDays=7) {{
    // aligned array; ignores nulls; at least 1 numeric in window needed
    const out = new Array(values.length).fill(null);
    let window = [];
    let windowSum = 0;

    for (let i = 0; i < values.length; i++) {{
      const v = values[i];
      if (v !== null && Number.isFinite(v)) {{
        window.push(v);
        windowSum += v;
      }} else {{
        window.push(null);
      }}

      if (i >= windowDays) {{
        const old = window.shift();
        if (old !== null && Number.isFinite(old)) windowSum -= old;
      }}

      let count = 0;
      for (const w of window) {{
        if (w !== null && Number.isFinite(w)) count++;
      }}
      if (count > 0) out[i] = windowSum / count;
    }}
    return out;
  }}

  function applyView() {{
    let start = clampDate(startInput.value);
    let end   = clampDate(endInput.value);

    if (!start || !end) return;
    if (start > end) {{
      alert("Start date must be ≤ End date.");
      return;
    }}

    startInput.value = start;
    endInput.value = end;

    const kSal = Math.max(0, Number(kSalInput.value || 0));
    const kPr  = Math.max(0, Number(kPrInput.value || 0));
    const prAboveOnly = !!prAboveOnlyInput.checked;

    // Filter arrays to date window
    const dates = [];
    const sal = [];
    const pr = [];

    for (let i = 0; i < DATA.dates.length; i++) {{
      const d = DATA.dates[i];
      if (d < start || d > end) continue;
      dates.push(d);
      sal.push(DATA.salinity[i]);
      pr.push(DATA.precip[i]);
    }}

    const totalDays = dates.length;
    const salPresent = sal.filter(v => v !== null && Number.isFinite(v)).length;
    const prPresent  = pr.filter(v => v !== null && Number.isFinite(v)).length;

    const salVals = sal.filter(v => v !== null && Number.isFinite(v));
    const prVals  = pr.filter(v => v !== null && Number.isFinite(v));

    const salStats = meanStd(salVals);
    const prStats  = meanStd(prVals);

    const salLo = salStats.mean - kSal * salStats.std;
    const salHi = salStats.mean + kSal * salStats.std;

    const prLo  = prStats.mean  - kPr  * prStats.std;
    const prHi  = prStats.mean  + kPr  * prStats.std;

    // Highlight arrays
    const salHX = [], salHY = [];
    for (let i = 0; i < dates.length; i++) {{
      const v = sal[i];
      if (v === null || !Number.isFinite(v)) continue;
      if (Number.isFinite(salLo) && Number.isFinite(salHi) && (v < salLo || v > salHi)) {{
        salHX.push(dates[i]); salHY.push(v);
      }}
    }}

    const prHX = [], prHY = [];
    if (Number.isFinite(prStats.mean) && Number.isFinite(prStats.std)) {{
      for (let i = 0; i < dates.length; i++) {{
        const v = pr[i];
        if (v === null || !Number.isFinite(v)) continue;

        if (prAboveOnly) {{
          const thr = prStats.mean + kPr * prStats.std;
          if (v > thr) {{
            prHX.push(dates[i]); prHY.push(v);
          }}
        }} else {{
          if (Number.isFinite(prLo) && Number.isFinite(prHi) && (v < prLo || v > prHi)) {{
            prHX.push(dates[i]); prHY.push(v);
          }}
        }}
      }}
    }}

    // Rolling 7-day means
    const salRoll = rollingMean(sal, 7);
    const prRoll  = rollingMean(pr, 7);

    // Update text indicators
    salStatsEl.innerHTML =
      `Salinity avg: <b>${{fmt(salStats.mean)}}</b> | std: <b>${{fmt(salStats.std)}}</b> | coverage: <b>${{pct(salPresent, totalDays)}}</b> | highlight: <b>mean ± ${{kSal}}·σ</b>`;

    if (prAboveOnly) {{
      const thr = prStats.mean + kPr * prStats.std;
      prStatsEl.innerHTML =
        `Precip avg: <b>${{fmt(prStats.mean)}}</b> | std: <b>${{fmt(prStats.std)}}</b> | coverage: <b>${{pct(prPresent, totalDays)}}</b> | highlight: <b>above mean + ${{kPr}}·σ</b> (thr=${{fmt(thr)}})`;
    }} else {{
      prStatsEl.innerHTML =
        `Precip avg: <b>${{fmt(prStats.mean)}}</b> | std: <b>${{fmt(prStats.std)}}</b> | coverage: <b>${{pct(prPresent, totalDays)}}</b> | highlight: <b>mean ± ${{kPr}}·σ</b>`;
    }}

    // Update Plotly traces by fixed indices
    Plotly.restyle(plotId, {{ x: [dates], y: [sal] }}, [0]);
    Plotly.restyle(plotId, {{ x: [salHX], y: [salHY] }}, [1]);
    Plotly.restyle(plotId, {{ x: [dates], y: [salRoll] }}, [2]);

    Plotly.restyle(plotId, {{ x: [dates], y: [pr] }}, [3]);
    Plotly.restyle(plotId, {{ x: [prHX], y: [prHY] }}, [4]);
    Plotly.restyle(plotId, {{ x: [dates], y: [prRoll] }}, [5]);

    Plotly.relayout(plotId, {{
      "xaxis.range": [start, end],
      "yaxis.autorange": true,
      "yaxis2.autorange": true
    }});
  }}

  applyBtn.addEventListener("click", applyView);
  resetBtn.addEventListener("click", () => {{
    startInput.value = dataStart;
    endInput.value = dataEnd;
    kSalInput.value = "1.0";
    kPrInput.value = "1.0";
    prAboveOnlyInput.checked = true;
    applyView();
  }});

  window.addEventListener("load", () => {{
    applyView();
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
    requested_end = date.today()
    requested_start = requested_end - timedelta(days=365 * 5)

    print(f"Requested: {requested_start} to {requested_end}")

    # Fetch continuous in chunks
    sal_cont = fetch_continuous_values_chunked(
        PIER26_MONITORING_LOCATION_ID,
        SALINITY_PCODE,
        requested_start,
        requested_end,
        statistic_id=CONT_STAT_ID,
        chunk_days=CHUNK_DAYS,
    )

    pr_cont = fetch_continuous_values_chunked(
        PIER26_MONITORING_LOCATION_ID,
        PRECIP_PCODE,
        requested_start,
        requested_end,
        statistic_id=CONT_STAT_ID,
        chunk_days=CHUNK_DAYS,
    )

    if pr_cont.empty:
      print("Precip is EMPTY for requested window.")
    else:
      print("Precip first:", pr_cont["time"].min())
      print("Precip last: ", pr_cont["time"].max())


    # Aggregate to daily
    sal_daily = to_daily_mean(sal_cont)
    pr_daily = to_daily_total(pr_cont)

    # Combine
    daily = pd.DataFrame(
        {
            "salinity": sal_daily,
            "precip_in": pr_daily,
        }
    ).sort_index()
    daily.index.name = "date_utc"

    # Defensive clip to last 5 years
    daily = daily.loc[
        (daily.index >= pd.to_datetime(requested_start, utc=True))
        & (daily.index <= pd.to_datetime(requested_end, utc=True))
    ]

    # Save CSV
    csv_path = BASE_DIR / "pier26_last5y_salinity_precip_daily.csv"
    daily.to_csv(csv_path)

    # Write dashboard as index.html (GitHub Pages default)
    html_path = BASE_DIR / "index.html"
    build_dashboard_html(
        daily=daily,
        output_html=str(html_path),
        title="Pier 26 (USGS-01376520): Salinity + Precipitation — last 5 years",
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved dashboard: {html_path}")
    print(f"Open in browser: open \"{html_path}\"")


if __name__ == "__main__":
    main()

    print("Done!")

