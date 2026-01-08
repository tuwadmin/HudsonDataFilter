
"""
Goal
----
Fetch the last 3 years of *precipitation* + *salinity* statistics for the
Hudson River Park Pier 26 area (USGS HRECOS gage is site 01376520). :contentReference[oaicite:0]{index=0}

Notes
-----
- USGS parameter code for total precipitation is 00045. :contentReference[oaicite:1]{index=1}
- USGS has a salinity parameter code 70386 (salinity computed from specific conductance). :contentReference[oaicite:2]{index=2}
- This script uses the USGS *modernized* Water Data OGC APIs for time series (daily/continuous)
  to pull data, then computes summary stats over the last 3 years.
  (The Swagger UI for the Statistics API v0 is JS-rendered and not reliably machine-readable
  in some environments, so this approach is robust and reproducible.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd


OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "hudson-pier26-stats-script/1.0"})


# ----------------------------
# Helpers
# ----------------------------
def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _iso(d: date) -> str:
    return d.isoformat()


def _pct(series: pd.Series, q: float) -> float:
    # q in [0, 1]
    return float(series.quantile(q, interpolation="linear"))


def summarize_series(s: pd.Series) -> Dict[str, Any]:
    s = s.dropna()
    if s.empty:
        return {"count": 0}

    return {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "p05": _pct(s, 0.05),
        "p10": _pct(s, 0.10),
        "p25": _pct(s, 0.25),
        "median": float(s.median()),
        "p75": _pct(s, 0.75),
        "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
    }


# ----------------------------
# Step 1: Define the Pier 26 (HRECOS) monitoring location
# ----------------------------
PIER26_MONITORING_LOCATION_ID = "USGS-01376520"  # NWIS site no 01376520; labeled Pier 25/26 in NWIS/HRECOS. :contentReference[oaicite:3]{index=3}

# target parameter codes
PRECIP_PCODE = "00045"   # Precipitation, total :contentReference[oaicite:4]{index=4}
SALINITY_PCODE_HINTS = {"70386"}  # common “salinity computed from specific conductance” code :contentReference[oaicite:5]{index=5}


# ----------------------------
# Step 2: Discover which parameter codes are actually available at Pier 26
#         using time-series metadata (so we don't guess wrong).
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
    """
    Returns mapping: parameter_code -> list of time series metadata records
    """
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for feat in ts_meta_features:
        props = feat.get("properties", {})
        pcode = str(props.get("parameter_code", "")).zfill(5) if props.get("parameter_code") else None
        if not pcode:
            continue
        mapping.setdefault(pcode, []).append(props)
    return mapping


# ----------------------------
# Step 3: Pull DAILY values for last 3 years if available.
#         If DAILY isn't available for a parameter, fall back to CONTINUOUS and aggregate to daily mean.
# ----------------------------
def fetch_daily_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = "00003",  # daily mean is commonly 00003 in USGS daily values systems
) -> pd.DataFrame:
    """
    Returns dataframe with columns: ['time', 'value'] (time is pandas datetime64[ns, UTC])
    """
    url = f"{OGC_BASE}/collections/daily/items"
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": parameter_code,
        "statistic_id": statistic_id,
        "time": f"{_iso(start)}/{_iso(end)}",
        "f": "json",
        "limit": 10000,
    }
    out = _get_json(url, params=params)
    feats = out.get("features", [])
    rows = []
    for f in feats:
        p = f.get("properties", {})
        t = p.get("time")
        v = p.get("value")
        if t is None or v is None:
            continue
        rows.append((pd.to_datetime(t, utc=True), float(v)))
    df = pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")
    return df


def fetch_continuous_values(
    monitoring_location_id: str,
    parameter_code: str,
    start: date,
    end: date,
    statistic_id: str = "00011",  # continuous values often use 00011 (instantaneous) in USGS modernized services
) -> pd.DataFrame:
    """
    Returns dataframe with columns: ['time', 'value'] (time is pandas datetime64[ns, UTC])
    """
    url = f"{OGC_BASE}/collections/continuous/items"
    params = {
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": parameter_code,
        "statistic_id": statistic_id,
        "time": f"{_iso(start)}/{_iso(end)}",
        "f": "json",
        "limit": 10000,
    }

    # The continuous endpoint may paginate with "links" / next; handle pagination safely.
    rows = []
    next_url = url
    next_params = params

    while True:
        out = _get_json(next_url, params=next_params)
        feats = out.get("features", [])
        for f in feats:
            p = f.get("properties", {})
            t = p.get("time")
            v = p.get("value")
            if t is None or v is None:
                continue
            rows.append((pd.to_datetime(t, utc=True), float(v)))

        # Look for a "next" link (OGC APIs typically provide this)
        next_link = None
        for link in out.get("links", []) or []:
            if link.get("rel") == "next" and link.get("href"):
                next_link = link["href"]
                break
        if not next_link:
            break

        # after the first request, follow the next link directly
        next_url = next_link
        next_params = None

    df = pd.DataFrame(rows, columns=["time", "value"]).sort_values("time")
    return df


def to_daily_mean(df: pd.DataFrame) -> pd.Series:
    """
    Convert a time-indexed value dataframe to daily mean series.
    """
    if df.empty:
        return pd.Series(dtype="float64")
    s = df.set_index("time")["value"]
    return s.resample("D").mean()


# ----------------------------
# Step 4: If Pier 26 doesn't have precipitation, find a nearby monitoring location with precip (00045).
#         (Many water-quality sites won’t measure rainfall.)
# ----------------------------
def get_monitoring_location(monitoring_location_id: str) -> Dict[str, Any]:
    """
    Robust approach for this API: query the monitoring-locations items list using
    a supported queryable field (id), then return the first feature.
    """
    url = f"{OGC_BASE}/collections/monitoring-locations/items"
    params = {
        "id": monitoring_location_id,  # queryables include `id` :contentReference[oaicite:1]{index=1}
        "f": "json",
        "limit": 1,
    }
    out = _get_json(url, params=params)
    feats = out.get("features", [])
    if not feats:
        raise ValueError(f"Monitoring location not found for id={monitoring_location_id}")
    return feats[0]



def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearby_precip_site(
    pier_feat: Dict[str, Any],
    bbox_pad_deg: float = 0.25,
    max_candidates: int = 200,
) -> Optional[str]:
    """
    Search nearby monitoring locations, then pick the nearest that has a 00045 time series.
    Returns monitoring_location_id or None.
    """
    geom = pier_feat.get("geometry", {})
    if geom.get("type") != "Point":
        return None
    lon, lat = geom.get("coordinates", [None, None])
    if lat is None or lon is None:
        return None

    bbox = f"{lon - bbox_pad_deg},{lat - bbox_pad_deg},{lon + bbox_pad_deg},{lat + bbox_pad_deg}"

    # Pull monitoring locations in bbox (cap at max_candidates)
    url = f"{OGC_BASE}/collections/monitoring-locations/items"
    params = {"bbox": bbox, "f": "json", "limit": max_candidates}
    out = _get_json(url, params=params)

    candidates = []
    for f in out.get("features", []):
        mid = f.get("properties", {}).get("monitoring_location_id")
        g = f.get("geometry", {})
        if not mid or g.get("type") != "Point":
            continue
        clon, clat = g.get("coordinates", [None, None])
        if clat is None or clon is None:
            continue
        candidates.append((mid, clat, clon))

    # For each candidate, check if there is time-series metadata for precip 00045
    best = None
    best_dist = None
    for mid, clat, clon in candidates:
        try:
            meta = get_time_series_metadata(mid)
        except Exception:
            continue
        available = extract_available_parameter_codes(meta)
        if PRECIP_PCODE in available:
            dist = haversine_km(lat, lon, clat, clon)
            if best is None or dist < best_dist:
                best = mid
                best_dist = dist

    return best


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    end = date.today()
    start = end - timedelta(days=365 * 3)

    # Get Pier 26 monitoring location feature (for coordinates / nearby search)
    pier_feat = get_monitoring_location(PIER26_MONITORING_LOCATION_ID)

    # Discover available parameter codes at Pier 26
    pier_ts_meta = get_time_series_metadata(PIER26_MONITORING_LOCATION_ID)
    pier_params = extract_available_parameter_codes(pier_ts_meta)

    # Resolve salinity parameter code at Pier 26:
    # prefer 70386 if present; otherwise look for any parameter with "salinity" in description/name.
    salinity_pcode = None
    if SALINITY_PCODE_HINTS & set(pier_params.keys()):
        salinity_pcode = sorted(SALINITY_PCODE_HINTS & set(pier_params.keys()))[0]
    else:
        # heuristic search through metadata descriptions
        for pcode, records in pier_params.items():
            text = " ".join(
                str(r.get("parameter_name", "")) + " " + str(r.get("parameter_description", ""))
                for r in records
            ).lower()
            if "salinity" in text:
                salinity_pcode = pcode
                break

    if not salinity_pcode:
        raise RuntimeError(
            "Could not find a salinity-related parameter code for Pier 26 from time-series metadata. "
            "Inspect pier_params keys or print pier_ts_meta for details."
        )

    # SALINITY: try daily first; fall back to continuous -> daily mean
    try:
        sal_daily_df = fetch_daily_values(PIER26_MONITORING_LOCATION_ID, salinity_pcode, start, end)
        sal_series = sal_daily_df.set_index("time")["value"]
        if sal_series.empty:
            raise ValueError("Empty daily salinity series")
    except Exception:
        sal_cont_df = fetch_continuous_values(PIER26_MONITORING_LOCATION_ID, salinity_pcode, start, end)
        sal_series = to_daily_mean(sal_cont_df)

    sal_stats = summarize_series(sal_series)

    # PRECIPITATION: check if Pier 26 even has it; if not, find nearby precip site with 00045.
    precip_location_id = PIER26_MONITORING_LOCATION_ID
    if PRECIP_PCODE not in pier_params:
        nearby = find_nearby_precip_site(pier_feat)
        if not nearby:
            raise RuntimeError(
                "Pier 26 does not appear to have precipitation (00045), and no nearby precip site was found "
                "in the search bbox. Try increasing bbox_pad_deg or using a known NOAA/NWS station instead."
            )
        precip_location_id = nearby

    # PRECIP: daily preferred; fall back to continuous -> daily mean
    try:
        pr_daily_df = fetch_daily_values(precip_location_id, PRECIP_PCODE, start, end)
        pr_series = pr_daily_df.set_index("time")["value"]
        if pr_series.empty:
            raise ValueError("Empty daily precipitation series")
    except Exception:
        pr_cont_df = fetch_continuous_values(precip_location_id, PRECIP_PCODE, start, end)
        pr_series = to_daily_mean(pr_cont_df)

    pr_stats = summarize_series(pr_series)

    print("=== Last 3 years summary statistics ===")
    print(f"Window: {start} to {end}\n")

    print(f"Salinity site: {PIER26_MONITORING_LOCATION_ID} | parameter_code={salinity_pcode}")
    print(sal_stats, "\n")

    print(f"Precip site:   {precip_location_id} | parameter_code={PRECIP_PCODE}")
    print(pr_stats, "\n")

    # Optional: save daily series to CSV
    out = pd.DataFrame(
        {
            "salinity": sal_series,
            "precip_total": pr_series,
        }
    )
    out.index.name = "date_utc"
    out.to_csv("pier26_last3y_salinity_precip_daily.csv")
    print("Saved daily time series to pier26_last3y_salinity_precip_daily.csv")

