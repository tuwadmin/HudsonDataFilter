"""
Pier 26 Dashboard: USGS Salinity + NOAA Precipitation (inches)

Outputs:
- pier26_last5y_salinity_precip_daily.csv (merged daily cache)
- noaa_prcp_daily.csv (NOAA-only cache)
- index.html (interactive dashboard)

Install deps:
  python3 -m pip install requests pandas plotly

Run:
  export NOAA_TOKEN="YOUR_TOKEN"
  python3 data.py
  open index.html
"""

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple

import os
import json
import time

import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

WINDOW_YEARS = 5

# USGS OGC API
OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"
PIER26_MONITORING_LOCATION_ID = "USGS-01376520"
SALINITY_PCODE = "90860"
USGS_CONT_STAT_ID = "00011"

# Smaller chunks/pages reduce stalls + rate-limits
USGS_CHUNK_DAYS = 30
USGS_LIMIT_PER_PAGE = 2000

# NOAA CDO API (GHCND daily)
NOAA_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NOAA_STATION_ID = "GHCND:USW00094728"  # Central Park precip proxy
NOAA_LIMIT = 200  # per page

# Cache + output
DAILY_CACHE_CSV = BASE_DIR / "pier26_last5y_salinity_precip_daily.csv"
NOAA_CACHE_CSV = BASE_DIR / "noaa_prcp_daily.csv"
DASHBOARD_HTML = BASE_DIR / "index.html"


# ----------------------------
# Sessions with retry
# ----------------------------
def _make_retry(total: int) -> Retry:
    return Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )


USGS_SESSION = requests.Session()
USGS_SESSION.headers.update({"User-Agent": "pier26-dashboard/1.0"})
USGS_SESSION.mount("https://", HTTPAdapter(max_retries=_make_retry(8)))

NOAA_SESSION = requests.Session()
NOAA_SESSION.mount("https://", HTTPAdapter(max_retries=_make_retry(6)))


# ----------------------------
# Utilities
# ----------------------------
def _date_range_iso(start: date, end: date) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


def _get_json_usgs(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = USGS_SESSION.get(url, params=params, timeout=(10, 180))

    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                wait_s = int(float(ra))
                print(f"USGS 429: sleeping {wait_s}s then retrying...")
                time.sleep(wait_s)
                r = USGS_SESSION.get(url, params=params, timeout=(10, 180))
            except Exception:
                pass

    r.raise_for_status()
    return r.json()


def _get_json_noaa(endpoint: str, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    All NOAA request handling lives here (prevents 'r not defined' errors).
    Includes status timing + 429 Retry-After respect.
    """
    url = f"{NOAA_BASE}{endpoint}"
    t0 = time.time()

    r = NOAA_SESSION.get(url, headers={"token": token}, params=params, timeout=(10, 180))
    dt = time.time() - t0
    print(f"NOAA status={r.status_code} in {dt:.1f}s | offset={params.get('offset')} limit={params.get('limit')}")

    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                wait_s = int(float(ra))
                print(f"NOAA 429: sleeping {wait_s}s then retrying...")
                time.sleep(wait_s)
                r = NOAA_SESSION.get(url, headers={"token": token}, params=params, timeout=(10, 180))
                print(f"NOAA retry status={r.status_code}")
            except Exception:
                pass

    r.raise_for_status()
    return r.json()


# ----------------------------
# USGS: continuous -> daily mean
# ----------------------------
def fetch_usgs_continuous_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = USGS_CONT_STAT_ID,
) -> pd.DataFrame:
    url = f"{OGC_BASE}/collections/continuous/items"
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": parameter_code,
        "statistic_id": statistic_id,
        "time": _date_range_iso(start, end),
        "f": "json",
        "limit": USGS_LIMIT_PER_PAGE,
    }

    rows: List[Tuple[pd.Timestamp, float]] = []
    next_url: Optional[str] = url
    next_params: Optional[Dict[str, Any]] = params

    while next_url:
        out = _get_json_usgs(next_url, params=next_params)

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
            next_params = None
            time.sleep(0.15)  # pacing
        else:
            next_url = None

    return pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")


def fetch_usgs_continuous_values_chunked(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = USGS_CONT_STAT_ID,
    chunk_days: int = USGS_CHUNK_DAYS,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    cur = start
    while cur <= end:
        cur_end = min(end, cur + timedelta(days=chunk_days))
        df_part = fetch_usgs_continuous_values(
            monitoring_location_id=monitoring_location_id,
            parameter_code=parameter_code,
            start=cur,
            end=cur_end,
            statistic_id=statistic_id,
        )
        print(f"Fetched USGS CONT {parameter_code} {cur} to {cur_end}: {len(df_part)} rows")
        if not df_part.empty:
            parts.append(df_part)
        cur = cur_end + timedelta(days=1)

    if not parts:
        return pd.DataFrame(columns=["time", "value"])

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return df.reset_index(drop=True)


def to_daily_mean(df: pd.DataFrame, name: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64", name=name)
    s = df.set_index("time")["value"].resample("D").mean()
    s.index = s.index.date
    s.name = name
    return s


# ----------------------------
# NOAA: daily PRCP (inches) + cache
# ----------------------------
def fetch_noaa_ghcnd_prcp_daily_in(
    station_id: str,
    start: date,
    end: date,
    token: str,
) -> pd.Series:
    """
    GHCND PRCP, units=standard (inches). Query year-by-year, paged.
    """
    all_rows: List[Dict[str, Any]] = []
    y = start.year

    while y <= end.year:
        chunk_start = date(y, 1, 1)
        chunk_end = date(y, 12, 31)
        if y == start.year:
            chunk_start = start
        if y == end.year:
            chunk_end = end

        print(f"Fetching NOAA PRCP {station_id} {chunk_start} to {chunk_end} ...")

        limit = NOAA_LIMIT
        offset = 1

        while True:
            out = _get_json_noaa(
                "/data",
                token=token,
                params={
                    "datasetid": "GHCND",
                    "datatypeid": "PRCP",
                    "stationid": station_id,
                    "startdate": chunk_start.isoformat(),
                    "enddate": chunk_end.isoformat(),
                    "units": "standard",  # inches
                    "limit": limit,
                    "offset": offset,
                },
            )

            rows = out.get("results", [])
            if not rows:
                break

            all_rows.extend(rows)
            time.sleep(0.2)

            meta = out.get("metadata", {}).get("resultset", {}) or {}
            count = int(meta.get("count", 0) or 0)
            if offset + limit > count:
                break
            offset += limit

        y += 1

    if not all_rows:
        return pd.Series(dtype="float64", name="precip_in")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    s = df.set_index("date")["value"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "precip_in"
    return s


def load_or_fetch_noaa_precip(token: str, start: date, end: date) -> pd.Series:
    if NOAA_CACHE_CSV.exists():
        print(f"Loading cached NOAA precip: {NOAA_CACHE_CSV.name}")
        tmp = pd.read_csv(NOAA_CACHE_CSV)
        if "date" not in tmp.columns:
            tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        if "precip_in" not in tmp.columns:
            for c in tmp.columns:
                if "precip" in c.lower():
                    tmp = tmp.rename(columns={c: "precip_in"})
                    break
        s = tmp.set_index("date")["precip_in"].astype(float).sort_index()
        s = s.loc[(s.index >= start) & (s.index <= end)]
        s.name = "precip_in"
        return s

    print("NOAA cache not found — downloading NOAA PRCP...")
    s = fetch_noaa_ghcnd_prcp_daily_in(
        station_id=NOAA_STATION_ID,
        start=start,
        end=end,
        token=token,
    )
    pd.DataFrame({"date": s.index, "precip_in": s.values}).to_csv(NOAA_CACHE_CSV, index=False)
    print(f"Saved NOAA cache: {NOAA_CACHE_CSV.name}")
    return s


# ----------------------------
# Daily cache (robust)
# ----------------------------
def load_daily_cache(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "date" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    daily = raw.set_index("date").sort_index()
    daily.index.name = "date"

    if "salinity" not in daily.columns:
        raise ValueError("Daily cache missing required column: salinity")
    if "precip_in" not in daily.columns:
        for c in daily.columns:
            if "precip" in c.lower():
                daily = daily.rename(columns={c: "precip_in"})
                break
    return daily


def save_daily_cache(path: Path, daily: pd.DataFrame) -> None:
    out = daily.copy()
    out.index.name = "date"
    out.reset_index().to_csv(path, index=False)


# ----------------------------
# Dashboard
# ----------------------------
def build_dashboard_html(daily: pd.DataFrame, output_html: str, title: str) -> str:
    if daily.empty:
        raise ValueError("No daily data to plot.")

    data_start = daily.index.min().isoformat()
    data_end = daily.index.max().isoformat()

    dates = [d.isoformat() for d in daily.index]
    sal = [None if pd.isna(v) else float(v) for v in daily["salinity"].tolist()]

    if "precip_in" in daily.columns:
        pr = [None if pd.isna(v) else float(v) for v in daily["precip_in"].tolist()]
    else:
        pr = [None for _ in dates]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("Salinity (daily mean)", "Precipitation (NOAA GHCND PRCP, inches)"),
    )

    # Salinity: line + highlighted markers + 7d mean
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Salinity", opacity=0.85), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="markers", name="Salinity (highlighted)",
            marker=dict(size=7, symbol="circle-open"), opacity=0.95
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Salinity (7d mean)", opacity=0.9), row=1, col=1)

    # Precip: bars + highlighted bars + 7d mean
    fig.add_trace(go.Bar(x=[], y=[], name="Precip (in)", opacity=0.45), row=2, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name="Precip (highlighted)", opacity=0.9), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Precip (7d mean)", opacity=0.9), row=2, col=1)

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=980,
        margin=dict(l=60, r=30, t=90, b=60),
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Salinity", row=1, col=1)
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
      Highlights use <b>mean ± (k·σ)</b> over the displayed interval (precip can optionally highlight only above avg).
      Rolling overlays are <b>7-day means</b>.
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
    const std = Math.sqrt(ss / n);
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

    const salRoll = rollingMean(sal, 7);
    const prRoll  = rollingMean(pr, 7);

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
    requested_start = requested_end - timedelta(days=365 * WINDOW_YEARS)
    print(f"Requested window: {requested_start} to {requested_end}")

    noaa_token = os.environ.get("NOAA_TOKEN")
    if not noaa_token:
        raise RuntimeError(
            "NOAA_TOKEN is not set.\n"
            "In Terminal run:\n"
            "  export NOAA_TOKEN='YOUR_TOKEN_HERE'\n"
            "Then re-run the script."
        )

    # Fast path: load merged cache if present
    if DAILY_CACHE_CSV.exists():
        print(f"Loading cached daily data: {DAILY_CACHE_CSV.name}")
        daily = load_daily_cache(DAILY_CACHE_CSV)
        daily = daily.loc[(daily.index >= requested_start) & (daily.index <= requested_end)]
    else:
        # USGS salinity (continuous -> daily mean)
        sal_cont = fetch_usgs_continuous_values_chunked(
            monitoring_location_id=PIER26_MONITORING_LOCATION_ID,
            parameter_code=SALINITY_PCODE,
            start=requested_start,
            end=requested_end,
            statistic_id=USGS_CONT_STAT_ID,
            chunk_days=USGS_CHUNK_DAYS,
        )
        sal_daily = to_daily_mean(sal_cont, name="salinity")

        # NOAA precip (with cache)
        noaa_precip = load_or_fetch_noaa_precip(
            token=noaa_token,
            start=requested_start,
            end=requested_end,
        )
        print(f"NOAA precip rows (daily): {len(noaa_precip)}")

        daily = pd.DataFrame({"salinity": sal_daily}).join(noaa_precip, how="outer").sort_index()
        daily.index.name = "date"
        daily = daily.loc[(daily.index >= requested_start) & (daily.index <= requested_end)]

        save_daily_cache(DAILY_CACHE_CSV, daily)
        print(f"Saved daily cache CSV: {DAILY_CACHE_CSV.name}")

    build_dashboard_html(
        daily=daily,
        output_html=str(DASHBOARD_HTML),
        title=f"Pier 26: USGS Salinity + NOAA Precip (inches) — last {WINDOW_YEARS} years",
    )
    print(f"Saved dashboard: {DASHBOARD_HTML}")
    print(f"Open in browser: open \"{DASHBOARD_HTML}\"")


if __name__ == "__main__":
    main()

    print("Done!")

