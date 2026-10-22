"""Build harmonised lithology classes and stratified sample points.

Sampling rules:
* keep only polygons with an interpretable lithology class,
* drop polygons smaller than MIN_AREA_KM2 (their 1:1M outlines are unreliable),
* erode each polygon by BUFFER_M to avoid mixed pixels near contacts,
* sample a random number of points per polygon proportional to its area,
  with a per-class cap so no single unit dominates the training set.
"""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CRS_UTM = 32754  # UTM zone 54S covers both study regions

REGION_BBOX = {
    "mount_isa": (139.40, -20.80, 139.70, -20.55),
    "flinders": (138.50, -31.50, 138.80, -31.25),
}

# Region-specific sampling settings.  The Flinders Ranges have many small
# carbonate/metasedimentary outcrops, so we sample them more densely.
REGION_PARAMS = {
    "mount_isa": dict(
        min_area_km2=1.0,
        buffer_m=100.0,
        points_per_km2=6.0,
        max_points_per_class=600,
        # keep samples away from the Mount Isa city/mine footprint
        exclude_center_km=(139.49, -20.73, 6.0),
    ),
    "flinders": dict(
        min_area_km2=0.3,
        buffer_m=60.0,
        points_per_km2=8.0,
        max_points_per_class=600,
        exclude_center_km=None,
    ),
}
MIN_POINTS_PER_POLYGON = 4
RNG = np.random.default_rng(20260504)


def harmonise(lith: str | None) -> str | None:
    """Map the detailed GA lithology description to a broad, trainable class."""
    if not lith:
        return None
    s = lith.lower()
    if "regolith" in s:
        return "regolith"
    if "carbonate" in s:
        return "carbonate"
    if "fault" in s or "shear" in s:
        return None
    if "ultramafic" in s or "mafic" in s:
        return "mafic_igneous"
    if "felsic" in s and "intrusive" in s:
        return "felsic_intrusive"
    if "intermediate" in s and "intrusive" in s:
        return "felsic_intrusive"
    if "volcanic" in s:
        return "felsic_volcanic"
    if "metasedimentary" in s or "high grade metamorphic" in s:
        return "metasedimentary"
    if (
        "arenite" in s
        or "rudite" in s
        or "argillaceous" in s
        or "detrital sediment" in s
        or "sedimentary siliciclastic" in s
    ):
        return "clastic_sedimentary"
    return None


def sample_points_inside(polygon, n: int) -> list[Point]:
    """Rejection sampling of points inside a (multi)polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    pts: list[Point] = []
    attempts = 0
    max_attempts = max(500, n * 50)
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        p = Point(RNG.uniform(minx, maxx), RNG.uniform(miny, maxy))
        if polygon.contains(p):
            pts.append(p)
    return pts


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)
    r = 6371.0
    dp = lat2 - lat1
    dl = lon2 - lon1
    a = np.sin(dp / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_region(region: str) -> pd.DataFrame:
    cfg = REGION_PARAMS[region]
    gj = gpd.read_file(DATA / f"geology_{region}.geojson")
    gj = gj.set_crs(4326, allow_override=True).to_crs(CRS_UTM)
    gj["lith_class"] = gj["lithology"].map(harmonise)
    gj = gj[gj["lith_class"].notna()].copy()

    # The ArcGIS query returns complete polygons that merely intersect the
    # request envelope, so clip them to the actual study area before sampling.
    from shapely.geometry import box

    lon0, lat0, lon1, lat1 = REGION_BBOX[region]
    tf = Transformer.from_crs(4326, CRS_UTM, always_xy=True)
    x0, y0 = tf.transform(lon0, lat0)
    x1, y1 = tf.transform(lon1, lat1)
    clip_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    gj["geometry"] = gj.geometry.intersection(clip_box)
    gj = gj[~gj.geometry.is_empty].copy()

    gj["area_km2"] = gj.geometry.area / 1e6
    gj = gj[gj["area_km2"] >= cfg["min_area_km2"]].copy()
    gj["geometry"] = gj.geometry.buffer(-cfg["buffer_m"])
    gj = gj[~gj.geometry.is_empty & gj.geometry.notna()].copy()
    gj["area_km2"] = gj.geometry.area / 1e6
    gj = gj[gj["area_km2"] >= 0.05].copy()

    rows: list[dict] = []
    for _, poly in gj.iterrows():
        n = int(
            max(
                MIN_POINTS_PER_POLYGON,
                min(400, math.ceil(poly["area_km2"] * cfg["points_per_km2"])),
            )
        )
        pts = sample_points_inside(poly.geometry, n)
        for p in pts:
            rows.append(
                dict(
                    region=region,
                    x_utm=p.x,
                    y_utm=p.y,
                    lith_class=poly["lith_class"],
                    lithology_raw=poly.get("lithology"),
                    unit_name=poly.get("name"),
                    polygon_area_km2=poly["area_km2"],
                )
            )

    df = pd.DataFrame(rows)
    # Cap per class: keep a random subset of the largest class contributions.
    kept = []
    for cls, grp in df.groupby("lith_class"):
        if len(grp) > cfg["max_points_per_class"]:
            grp = grp.sample(cfg["max_points_per_class"], random_state=42)
        kept.append(grp)
    df = pd.concat(kept, ignore_index=True)

    # Convert to WGS84 for satellite queries.
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x_utm, df.y_utm), crs=CRS_UTM
    ).to_crs(4326)
    df["lon"] = pts.geometry.x
    df["lat"] = pts.geometry.y

    if cfg["exclude_center_km"]:
        clon, clat, radius = cfg["exclude_center_km"]
        dist = haversine_km(df["lon"], df["lat"], clon, clat)
        df = df[dist > radius].reset_index(drop=True)
    return df


def main() -> None:
    frames = []
    for region in ("mount_isa", "flinders"):
        df = build_region(region)
        df.to_csv(DATA / f"samples_{region}.csv", index=False)
        print(f"\n== {region}: {len(df)} samples")
        print(df["lith_class"].value_counts().to_string())
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    allp.to_csv(DATA / "samples_all.csv", index=False)
    print(f"\ntotal samples: {len(allp)}")


if __name__ == "__main__":
    main()
