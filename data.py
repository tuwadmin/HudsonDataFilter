"""
Hudson River Park (Pier 26 / USGS-01376520) — last 3 years precipitation + salinity
Clean, refactored script that:
  1) pulls USGS OGC API continuous data (statistic_id=00011),
  2) aggregates to daily series:
       - Salinity: daily mean
       - Precip: daily total (auto-detect incremental vs cumulative)
  3) prints summary stats
  4) saves CSV
  5) generates an interactive HTML dashboard you can open in a browser

Install deps:
  /usr/local/bin/python3 -m pip install requests pandas plotly

Run:
  /usr/local/bin/python3 "/Users/samialemfadli/Desktop/HRECSO Data/data.py"

Outputs:
  - pier26_last3y_daily.csv
  - pier26_last3y_dashboard.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# Config
# ----------------------------
OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pier26-salinity-precip/1.0"})

PIER26_MONITORING_LOCATION_ID = "USGS-01376520"

# Parameter codes (from your metadata)
SALINITY_PCODE = "90860"  # Salinity, wu, at 25C (as discovered at this site)
PRECIP_PCODE = "00045"    # Precipitation (units: inches at this site)

# Continuous statistic id at this site
CONT_STAT_ID = "00011"


# ----------------------------
# HTTP helpers
# ----------------------------
def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _date_range_iso(start: date, end: date) -> str:
    # OGC 'time' parameter is typically start/end (inclusive/exclusive varies by collection).
    return f"{start.isoformat()}/{end.isoformat()}"


# ----------------------------
# USGS OGC API functions
# ----------------------------
def get_time_series_metadata(monitoring_location_id: str) -> List[Dict[str, Any]]:
    url = f"{OGC_BASE}/collections/time-series-metadata/items"
    params = {
        "monitoring_location_id": monitoring_location_id,
        "f": "json",
        "limit": 1000,
    }
    out = _get_json(url, params=params)
    return out.get("features", [])


def extract_available_parameter_codes(ts_meta_features: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for feat in ts_meta_features:
        props = feat.get("properties", {})
        pcode = props.get("parameter_code")
        if not pcode:
            continue
        pcode = str(pcode).zfill(5) if str(pcode).isdigit() else str(pcode)
        mapping.setdefault(pcode, []).append(props)
    return mapping


def get_monitoring_location_feature(monitoring_location_id: str) -> Dict[str, Any]:
    """
    Robust retrieval via queryables (some environments reject /items/{id}).
    """
    url = f"{OGC_BASE}/collections/monitoring-locations/items"
    params = {"id": monitoring_location_id, "f": "json", "limit": 1}
    out = _get_json(url, params=params)
    feats = out.get("features", [])
    if not feats:
        raise ValueError(f"Monitoring location not found: id={monitoring_location_id}")
    return feats[0]


def fetch_continuous_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = CONT_STAT_ID,
) -> pd.DataFrame:
    """
    Returns DataFrame columns: time (UTC), value (float)
    Handles pagination via 'next' links when present.
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

        # Find "next" link
        next_link = None
        for link in out.get("links", []) or []:
            if link.get("rel") == "next" and link.get("href"):
                next_link = link["href"]
                break

        if next_link:
            next_url = next_link
            next_params = None  # next link already contains query
        else:
            next_url = None

    df = pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")
    return df


# ----------------------------
# Aggregation helpers
# ----------------------------
def to_daily_mean(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index("time")["value"].resample("D").mean()


def to_daily_sum(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    return df.set_index("time")["value"].resample("D").sum()


def detect_precip_mode(df: pd.DataFrame) -> str:
    """
    Heuristic:
      - If values are almost always non-decreasing within a day and resets sometimes -> cumulative gauge
      - Otherwise -> incremental tips/amounts
    Returns: "cumulative" or "incremental"

    This is a best-effort heuristic based on typical precipitation sensor behavior.
    """
    if df.empty or df.shape[0] < 50:
        return "incremental"

    s = df.set_index("time")["value"].sort_index()

    # Sample a subset for speed if huge
    if s.shape[0] > 200_000:
        s = s.iloc[:: max(1, s.shape[0] // 200_000)]

    # Check within-day monotonicity (cumulative tends to be non-decreasing most of the time)
    by_day = s.groupby(s.index.floor("D"))
    nondec_ratios = []
    for _, day_vals in by_day:
        if day_vals.shape[0] < 5:
            continue
        diffs = day_vals.diff().dropna()
        if diffs.empty:
            continue
        nondec = (diffs >= 0).mean()
        nondec_ratios.append(float(nondec))
        if len(nondec_ratios) >= 30:
            break

    if not nondec_ratios:
        return "incremental"

    # If most sampled days are strongly non-decreasing, treat as cumulative
    if sum(r > 0.90 for r in nondec_ratios) / len(nondec_ratios) >= 0.70:
        return "cumulative"

    return "incremental"


def precip_daily_total(df: pd.DataFrame) -> pd.Series:
    """
    Returns daily precipitation totals in the gauge's unit (inches here).

    - If incremental: daily total = sum of values in day
    - If cumulative: daily total = (daily max - daily min), clipped at >= 0
    """
    if df.empty:
        return pd.Series(dtype="float64")

    mode = detect_precip_mode(df)
    s = df.set_index("time")["value"].sort_index()

    if mode == "incremental":
        return s.resample("D").sum()

    # cumulative
    daily_max = s.resample("D").max()
    daily_min = s.resample("D").min()
    daily = (daily_max - daily_min).clip(lower=0)
    return daily


# ----------------------------
# Stats helpers
# ----------------------------
def summarize_series(s: pd.Series) -> Dict[str, Any]:
    s = s.dropna()
    if s.empty:
        return {"count": 0}

    def pct(q: float) -> float:
        return float(s.quantile(q, interpolation="linear"))

    return {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "p05": pct(0.05),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": float(s.median()),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
    }

# ----------------------------
# Visualization (HTML)
# ----------------------------
def build_dashboard_html(
    daily_df: pd.DataFrame,
    title: str,
    salinity_col: str = "salinity",
    precip_col: str = "precip_in",
    output_html: str = "pier26_last3y_dashboard.html",
    default_start=None,
    default_end=None,
) -> str:

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Salinity (daily mean)", "Precipitation (daily total)"),
    )

    fig.add_trace(
        go.Scatter(
            x=daily_df.index,
            y=daily_df[salinity_col],
            mode="lines",
            name="Salinity",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=daily_df.index,
            y=daily_df[precip_col],
            name="Precip (in)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=800,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.update_yaxes(title_text="Salinity (sensor units)", row=1, col=1)
    fig.update_yaxes(title_text="Inches", row=2, col=1)

    # Range selector + slider
    range_selector = dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]
    )

    fig.update_xaxes(
        title_text="Date",
        rangeslider=dict(visible=True),
        rangeselector=range_selector,
        row=2,
        col=1,
    )
    fig.update_xaxes(rangeselector=range_selector, row=1, col=1)

    # ✅ Force initial view to the full dataset range (your last 3 years)
    if default_start is not None and default_end is not None:
        fig.update_xaxes(range=[default_start, default_end], row=1, col=1)
        fig.update_xaxes(range=[default_start, default_end], row=2, col=1)

    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html


# --- 2) IN main(), right before calling build_dashboard_html(...), add these lines ---

# html_path = build_dashboard_html(
#     daily_df=daily,
#     title="Pier 26 (USGS-01376520): Salinity & Precipitation — last 3 years",
#     output_html=str(BASE_DIR / "pier26_last3y_dashboard.html"),
#     default_start=full_start,
#     default_end=full_end,
# )

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    end = date.today()
    start = end - timedelta(days=365 * 3)

    # Confirm site exists and pull metadata (optional validation)
    site = get_monitoring_location_feature(PIER26_MONITORING_LOCATION_ID)
    site_name = site.get("properties", {}).get("monitoring_location_name", PIER26_MONITORING_LOCATION_ID)

    ts_meta = get_time_series_metadata(PIER26_MONITORING_LOCATION_ID)
    params_map = extract_available_parameter_codes(ts_meta)

    # Soft validation that these parameters exist at this site
    if SALINITY_PCODE not in params_map:
        raise RuntimeError(f"Salinity parameter {SALINITY_PCODE} not found at site {PIER26_MONITORING_LOCATION_ID}")
    if PRECIP_PCODE not in params_map:
        raise RuntimeError(f"Precip parameter {PRECIP_PCODE} not found at site {PIER26_MONITORING_LOCATION_ID}")

    # Fetch continuous series
    sal_cont = fetch_continuous_values(PIER26_MONITORING_LOCATION_ID, SALINITY_PCODE, start, end, statistic_id=CONT_STAT_ID)
    pr_cont = fetch_continuous_values(PIER26_MONITORING_LOCATION_ID, PRECIP_PCODE, start, end, statistic_id=CONT_STAT_ID)

    # Aggregate to daily
    sal_daily = to_daily_mean(sal_cont)
    pr_daily = precip_daily_total(pr_cont)

    daily = pd.DataFrame({
        "salinity": sal_daily,
        "precip_in": pr_daily,
    })
    daily.index.name = "date_utc"

    full_start = daily.index.min()
    full_end = daily.index.max()

    html_path = build_dashboard_html(
        daily_df=daily,
        title="Pier 26 (USGS-01376520): Salinity & Precipitation — last 3 years",
        output_html=str(BASE_DIR / "pier26_last3y_dashboard.html"),
        default_start=full_start,
        default_end=full_end,
    )


    # Summary stats
    sal_stats = summarize_series(daily["salinity"])
    pr_stats = summarize_series(daily["precip_in"])

    print("=== Last 3 years summary statistics ===")
    print(f"Site: {PIER26_MONITORING_LOCATION_ID} — {site_name}")
    print(f"Window: {start} to {end}\n")

    print(f"Salinity (pcode={SALINITY_PCODE})")
    print(sal_stats, "\n")

    print(f"Precipitation (pcode={PRECIP_PCODE}, unit=in)")
    print(pr_stats, "\n")

    # Save CSV
    csv_path = "pier26_last3y_daily.csv"
    daily.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # Save interactive dashboard HTML
    html_path = build_dashboard_html(
        daily_df=daily,
        title=f"Pier 26 (USGS-01376520): Salinity & Precipitation — last 3 years",
        output_html="pier26_last3y_dashboard.html",
    )
    print(f"Saved dashboard: {html_path}")
    print("Open it in your browser (double-click in Finder, or run: open pier26_last3y_dashboard.html)")


if __name__ == "__main__":
    main()
    

    print("Done!")

