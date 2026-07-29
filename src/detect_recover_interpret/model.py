"""
Seattle building energy: honest EUI model, planted gate.

MODEL
  target    log SiteEUIWN (kBtu/sf/yr), weather-normalized site energy intensity
  honest h  smooth (spline + ridge) in building attributes and location.
            Smoothness is not incidental: the null class of routing_audit_v2 3.3 is
            smooth single-branch surfaces, so the honest model must live in it.
  gate      f(x) = h(x) - 0.30 * 1[ b(x) >= 0.25 ]
            b = tract Black/Latino population share; tau = 0.25 (codebase default)

HARM DIRECTION
  A subtractive penalty in log-EUI multiplies predicted intensity by exp(-0.30) = 0.74.
  Buildings in gated tracts appear 26% MORE EFFICIENT than they are, so they fall below
  retrofit-program eligibility thresholds and are screened out of efficiency investment.
  The harm is exclusion from capital, not overcharge.

PROBE
  Perturbs LOCATION (lat, lon). b is recomputed at the perturbed point, so probes near a
  tract boundary cross the gate. This is the geographic reading of the audit: "what would
  this model say about this building slightly elsewhere?"
"""
import json
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from . import paths

TAU_B = 0.25
PENALTY = 0.30
GRID_N = 1600

NUMERIC_COLS = ["Latitude", "Longitude", "PropertyGFATotal", "YearBuilt",
                "NumberofFloors", "NumberofBuildings"]


def load_buildings(csv_path=None):
    csv_path = csv_path or paths.SEATTLE_EUI_CSV
    df = pd.read_csv(csv_path, low_memory=False)
    eui = "SiteEUIWN(kBtu/sf)"
    for c in [eui, "Latitude", "Longitude", "PropertyGFATotal", "YearBuilt",
              "NumberofFloors", "NumberofBuildings", "DataYear"]:
        # several numeric columns ship with thousands separators
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce")
    df = df[df["DataYear"] == df["DataYear"].max()]
    df = df[(df[eui] > 1) & (df[eui] < 1000)]
    df = df[df["Latitude"].between(47.4, 47.8) & df["Longitude"].between(-122.5, -122.2)]
    if "Demolished" in df:
        dem = df["Demolished"].astype("object").map(
            {True: 1, False: 0, "Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0)
        df = df[dem == 0]
    keep = ["OSEBuildingID", "Latitude", "Longitude", eui, "PropertyGFATotal",
            "YearBuilt", "NumberofFloors", "NumberofBuildings",
            "LargestPropertyUseType", "Neighborhood"]
    df = df[keep].dropna(subset=["Latitude", "Longitude", eui, "PropertyGFATotal",
                                 "YearBuilt"])
    df = df.rename(columns={eui: "eui"})
    df["NumberofFloors"] = df["NumberofFloors"].fillna(1).clip(1, 80)
    df["NumberofBuildings"] = df["NumberofBuildings"].fillna(1).clip(1, 30)
    df["LargestPropertyUseType"] = df["LargestPropertyUseType"].fillna("Unknown")
    df["log_eui"] = np.log(df["eui"].to_numpy(float))
    return df.reset_index(drop=True)


def build_b_raster(lat_lo, lat_hi, lon_lo, lon_hi, n=GRID_N, tracts_path=None):
    """Rasterize tract Black/Latino share so b(lat,lon) is an O(1) array lookup.
    Probing needs millions of b evaluations; point-in-polygon per query is far too slow."""
    tracts_path = tracts_path or paths.TRACT_DEMOGRAPHICS_GEOJSON
    gj = json.load(open(tracts_path))
    lons = np.linspace(lon_lo, lon_hi, n)
    lats = np.linspace(lat_lo, lat_hi, n)
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    B = np.full(pts.shape[0], np.nan)
    for ft in gj["features"]:
        v = ft["properties"].get("pct_black_latino")
        if v is None:
            continue
        geom = ft["geometry"]
        polys = [geom["coordinates"][0]] if geom["type"] == "Polygon" else \
                [p[0] for p in geom["coordinates"]]
        for ring in polys:
            ring = np.asarray(ring, float)[:, :2]
            if ring[:, 0].max() < lon_lo or ring[:, 0].min() > lon_hi:
                continue
            if ring[:, 1].max() < lat_lo or ring[:, 1].min() > lat_hi:
                continue
            m = MplPath(ring).contains_points(pts)
            B[m] = v
    return {"B": B.reshape(n, n), "lats": lats, "lons": lons}


def b_lookup(ras, lat, lon):
    lats, lons, B = ras["lats"], ras["lons"], ras["B"]
    i = np.clip(((lat - lats[0]) / (lats[-1] - lats[0]) * (len(lats) - 1)).astype(int),
                0, len(lats) - 1)
    j = np.clip(((lon - lons[0]) / (lons[-1] - lons[0]) * (len(lons) - 1)).astype(int),
                0, len(lons) - 1)
    out = B[i, j]
    return np.nan_to_num(out, nan=0.0)


def fit_honest(df, num_cols=None, n_knots=6, degree=3, ridge_alpha=2.0):
    """Smooth honest model: spline basis on numeric features + location, one-hot on use type."""
    num = num_cols or NUMERIC_COLS
    X = df[num + ["LargestPropertyUseType"]].copy()
    X["PropertyGFATotal"] = np.log(X["PropertyGFATotal"].clip(lower=1))
    y = df["log_eui"].to_numpy(float)

    ct = ColumnTransformer([
        ("num", make_pipeline(StandardScaler(),
                              SplineTransformer(n_knots=n_knots, degree=degree)), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
         ["LargestPropertyUseType"]),
    ])
    model = make_pipeline(ct, Ridge(alpha=ridge_alpha)).fit(X, y)
    return model, num


def make_predictors(model, num_cols, ras, tau_b=TAU_B, penalty=PENALTY):
    """Return honest and gated prediction functions over (lat, lon, static features)."""
    def _frame(lat, lon, base_row):
        d = {c: np.full(len(lat), base_row[c]) for c in num_cols
             if c not in ("Latitude", "Longitude")}
        d["Latitude"] = lat
        d["Longitude"] = lon
        d["PropertyGFATotal"] = np.log(max(float(base_row["PropertyGFATotal"]), 1.0)) \
            * np.ones(len(lat))
        d["LargestPropertyUseType"] = np.array([base_row["LargestPropertyUseType"]] * len(lat))
        return pd.DataFrame(d)[num_cols + ["LargestPropertyUseType"]]

    def honest(lat, lon, base_row):
        return model.predict(_frame(lat, lon, base_row))

    def gated(lat, lon, base_row):
        h = honest(lat, lon, base_row)
        return h - penalty * (b_lookup(ras, lat, lon) >= tau_b)

    return honest, gated
