"""
Additive decomposition of the honest model's location dependence.

The honest model is additive in the spline basis (SplineTransformer per column + Ridge),
so h(lat, lon | building) = h_base + g_lat(lat) + g_lon(lon). Precomputing g_lat and g_lon
on a fine grid turns each probe into two interpolations instead of a full model.predict
call, which is what makes probing at O(1e5-1e6) query volume tractable.

Shared by every Seattle audit script (current and legacy) — this used to live inside
exp_seattle_audit.py's v1 script but the location-terms trick and the metres-per-degree
constants are infrastructure, not part of that superseded protocol.
"""
import numpy as np
import pandas as pd

M_LAT = 111320.0
M_LON = 111320.0 * np.cos(np.deg2rad(47.61))  # Seattle latitude


def build_location_terms(model, num_cols, df, n=3000):
    """Exploit additivity: tabulate g_lat and g_lon, and h_base per building."""
    med = {c: float(np.median(df[c])) for c in num_cols}
    med["PropertyGFATotal"] = float(np.median(np.log(df["PropertyGFATotal"].clip(lower=1))))
    mode_use = df["LargestPropertyUseType"].mode()[0]

    def frame(lat, lon):
        d = {c: np.full(len(lat), med[c]) for c in num_cols}
        d["Latitude"] = lat; d["Longitude"] = lon
        d["LargestPropertyUseType"] = np.array([mode_use] * len(lat))
        return pd.DataFrame(d)[num_cols + ["LargestPropertyUseType"]]

    lat_g = np.linspace(df["Latitude"].min() - 0.02, df["Latitude"].max() + 0.02, n)
    lon_g = np.linspace(df["Longitude"].min() - 0.02, df["Longitude"].max() + 0.02, n)
    ref_lat, ref_lon = med["Latitude"], med["Longitude"]
    base = float(model.predict(frame(np.array([ref_lat]), np.array([ref_lon])))[0])
    g_lat = model.predict(frame(lat_g, np.full(n, ref_lon))) - base
    g_lon = model.predict(frame(np.full(n, ref_lat), lon_g)) - base

    # per-building constant = full honest prediction at its own location, minus loc terms
    X = df[num_cols + ["LargestPropertyUseType"]].copy()
    X["PropertyGFATotal"] = np.log(X["PropertyGFATotal"].clip(lower=1))
    h_all = model.predict(X)
    h_base = h_all - np.interp(df["Latitude"], lat_g, g_lat) \
                   - np.interp(df["Longitude"], lon_g, g_lon)
    return {"lat_g": lat_g, "lon_g": lon_g, "g_lat": g_lat, "g_lon": g_lon,
            "h_base": h_base.to_numpy() if hasattr(h_base, "to_numpy") else np.asarray(h_base)}
