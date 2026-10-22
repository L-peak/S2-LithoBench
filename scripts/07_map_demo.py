"""Demonstration map: predict lithology for every 20 m pixel of a small
window inside one study region and compare it with the reference polygons.

The features are built exactly as for the training samples: 5-date median of
cloud-masked Sentinel-2 bands, spectral indices, Copernicus DEM terrain
attributes, and 3x3 local texture.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds, intersection
from scipy.ndimage import uniform_filter

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

STAC_URL = "https://earth-search.aws.element84.com/v1"
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

REGIONS = {
    "mount_isa": dict(bbox=(139.40, -20.80, 139.70, -20.55), datetime="2024-04-01/2024-10-31"),
    "flinders": dict(bbox=(138.50, -31.50, 138.80, -31.25), datetime="2024-04-01/2024-10-31"),
}

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}
INDICES = ["ndvi", "iron_oxide", "ferrous", "clay", "swir_ratio", "brightness"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos",
           "dem_curvature", "dem_tpi", "dem_roughness"]
TEXTURE = ["tex_B04", "tex_B08", "tex_B11"]
CLASS_COLORS = {
    "regolith": "#d9d9d9",
    "clastic_sedimentary": "#f2c14e",
    "carbonate": "#4c9f70",
    "metasedimentary": "#7b6cf6",
    "mafic_igneous": "#d1495b",
    "felsic_intrusive": "#e07a5f",
    "felsic_volcanic": "#2e86ab",
}
VALID_SCL = np.array([4, 5, 7])


def harmonise(lith: str | None) -> str | None:
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
    if any(k in s for k in ["arenite", "rudite", "argillaceous", "detrital sediment",
                            "sedimentary siliciclastic"]):
        return "clastic_sedimentary"
    return None


GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    CPL_VSIL_CURL_CHUNK_SIZE="1048576",
    AWS_NO_SIGN_REQUEST="YES",
    VSI_CACHE="TRUE",
)


def ref_grid(item, bounds):
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(item.assets["scl"].href) as src:
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            win = intersection(win, Window(0, 0, src.width, src.height))
            return window_bounds(win, src.transform), src.window_transform(win), (int(win.height), int(win.width))


def read_band(asset_key, item, snapped, shape):
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(item.assets[asset_key].href) as src:
            win = from_bounds(*snapped, transform=src.transform).round_offsets().round_lengths()
            win = intersection(win, Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None
            return src.read(1, window=win, out_shape=shape, resampling=Resampling.bilinear)


def dem_tile_name(lat: int, lon: int) -> str:
    ns = "S" if lat < 0 else "N"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def terrain_for_window(region, snapped, shape, transform):
    """Read DEM tiles covering the window and reproject terrain layers."""
    lon0, lat0, lon1, lat1 = snapped[0], snapped[1], snapped[2], snapped[3]
    tf = Transformer.from_crs(32754, 4326, always_xy=True)
    llon0, llat0 = tf.transform(lon0, lat0)
    llon1, llat1 = tf.transform(lon1, lat1)

    from rasterio.merge import merge

    tiles = set()
    for lat in (math.floor(min(llat0, llat1)), math.floor(max(llat0, llat1))):
        for lon in (math.floor(min(llon0, llon1)), math.floor(max(llon0, llon1))):
            n = dem_tile_name(lat, lon)
            tiles.add(f"{DEM_BUCKET}/{n}/{n}.tif")

    srcs = []
    with rasterio.Env(**GDAL_ENV):
        for href in sorted(tiles):
            srcs.append(rasterio.open(href))
        mosaic, dem_transform = merge(srcs)
        for s in srcs:
            s.close()
    dem = mosaic[0].astype(np.float32)
    del mosaic

    lat_mean = (llat0 + llat1) / 2
    res_deg = abs(dem_transform.a)
    dx_m = res_deg * 111320.0 * math.cos(math.radians(lat_mean))
    dy_m = res_deg * 110540.0
    d_drow = np.gradient(dem, axis=0)
    d_dcol = np.gradient(dem, axis=1)
    dzdy = -d_drow / dy_m
    dzdx = d_dcol / dx_m
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.arctan2(-dzdx, -dzdy)
    d2x = np.gradient(dzdx, axis=1) / dx_m
    d2y = -np.gradient(dzdy, axis=0) / dy_m
    curv = d2x + d2y
    tpi = dem - uniform_filter(dem, size=15, mode="nearest")
    m5 = uniform_filter(dem, size=5, mode="nearest")
    rough = np.sqrt(np.maximum(uniform_filter(dem**2, size=5, mode="nearest") - m5**2, 0))

    layers = {
        "dem_elev": dem,
        "dem_slope": slope,
        "dem_aspect_sin": np.sin(aspect),
        "dem_aspect_cos": np.cos(aspect),
        "dem_curvature": curv,
        "dem_tpi": tpi,
        "dem_roughness": rough,
    }
    from rasterio.warp import reproject

    out = {}
    for name, arr in layers.items():
        dst = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=arr,
            destination=dst,
            src_transform=dem_transform,
            src_crs="EPSG:4326",
            dst_transform=transform,
            dst_crs="EPSG:32754",
            resampling=Resampling.bilinear,
        )
        out[name] = dst
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="mount_isa")
    ap.add_argument("--size-km", type=float, default=8.0)
    ap.add_argument("--dates", type=int, default=5)
    args = ap.parse_args()

    region = args.region
    cfg = REGIONS[region]
    bundle = joblib.load(RESULTS / f"model_{region}.joblib")
    model, feats, classes = bundle["model"], bundle["features"], bundle["classes"]

    samples = pd.read_csv(DATA / f"features_{region}.csv")
    cx, cy = samples["x_utm"].mean(), samples["y_utm"].mean()
    half = args.size_km * 1000 / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)

    catalog = Client.open(STAC_URL)
    items = [
        it
        for it in catalog.search(
            collections=["sentinel-2-l2a"], bbox=cfg["bbox"], datetime=cfg["datetime"],
            query={"eo:cloud_cover": {"lt": 20}}, max_items=80,
        ).items()
        if it.bbox[0] <= cfg["bbox"][0] and it.bbox[1] <= cfg["bbox"][1]
        and it.bbox[2] >= cfg["bbox"][2] and it.bbox[3] >= cfg["bbox"][3]
    ]
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items = items[: args.dates]
    print("dates:", [f"{it.datetime:%Y-%m-%d}" for it in items])

    ref = ref_grid(items[0], bounds)
    snapped, transform, shape = ref
    print("grid:", shape)

    stacks = {b: [] for b in BANDS}
    scl_stack = []
    for it in items:
        offset = 0.0 if it.properties.get("earthsearch:boa_offset_applied", True) else 1000.0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {b: pool.submit(read_band, ASSETS[b], it, snapped, shape) for b in BANDS}
            arrs = {b: f.result() for b, f in futs.items()}
        scl = read_band("scl", it, snapped, shape)
        scl_stack.append(scl)
        for b in BANDS:
            a = arrs[b].astype(np.float32)
            a = (a - offset) / 10000.0 if offset else a / 10000.0
            stacks[b].append(a)

    valid = np.ones(shape, dtype=bool)
    for scl in scl_stack:
        valid &= np.isin(scl, VALID_SCL)
    n_valid = valid.sum(axis=0)
    keep = n_valid >= 3

    comp = {}
    for b in BANDS:
        stack = np.stack(stacks[b])
        stack[:, ~valid] = np.nan
        comp[b] = np.nanmedian(stack, axis=0)
        del stack
    comp = {b: np.where(keep, v, np.nan) for b, v in comp.items()}

    eps = 1e-6
    feats_map = dict(comp)
    feats_map["ndvi"] = (comp["B08"] - comp["B04"]) / (comp["B08"] + comp["B04"] + eps)
    feats_map["iron_oxide"] = comp["B04"] / (comp["B02"] + eps)
    feats_map["ferrous"] = comp["B11"] / (comp["B08"] + eps)
    feats_map["clay"] = comp["B11"] / (comp["B12"] + eps)
    feats_map["swir_ratio"] = (comp["B11"] - comp["B12"]) / (comp["B11"] + comp["B12"] + eps)
    feats_map["brightness"] = (comp["B02"] + comp["B03"] + comp["B04"]) / 3
    for b in ("B04", "B08", "B11"):
        m = uniform_filter(np.nan_to_num(comp[b], nan=np.nanmean(comp[b])), size=3)
        m2 = uniform_filter(np.nan_to_num(comp[b] ** 2, nan=np.nanmean(comp[b] ** 2)), size=3)
        feats_map[f"tex_{b}"] = np.sqrt(np.maximum(m2 - m**2, 0))
    feats_map.update(terrain_for_window(region, snapped, shape, transform))

    X = np.stack([feats_map[f].ravel() for f in feats], axis=1)
    ok = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], -1, dtype=int)
    if ok.any():
        pred[ok] = model.predict(X[ok])
    pred = pred.reshape(shape)

    # reference raster from the GA polygons
    import geopandas as gpd

    geo = gpd.read_file(DATA / f"geology_{region}.geojson").set_crs(4326, allow_override=True).to_crs(32754)
    geo["lith_class"] = geo["lithology"].map(harmonise)
    cls_to_int = {c: i for i, c in enumerate(classes)}
    shapes = [
        (geom, cls_to_int[c])
        for geom, c in zip(geo.geometry, geo["lith_class"])
        if c in cls_to_int
    ]
    ref_r = rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int16")
    ref_r = np.where(keep, ref_r, -1)
    pred = np.where(keep, pred, -1)

    np.save(RESULTS / f"map_pred_{region}.npy", pred)
    np.save(RESULTS / f"map_ref_{region}.npy", ref_r)

    cmap = ListedColormap([CLASS_COLORS.get(c, "#333333") for c in classes])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, arr, title in zip(axes, (ref_r, pred), ("Reference (GA 1:1M)", "Predicted")):
        masked = np.ma.masked_where(arr < 0, arr)
        ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=6,
                   markerfacecolor=CLASS_COLORS.get(c, "#333"), markeredgecolor="none", label=c)
        for c in classes
    ]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"{region}: {args.size_km:.0f} km × {args.size_km:.0f} km window", y=0.98)
    fig.savefig(FIG / f"fig8_map_{region}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"fig8_map_{region}.pdf", bbox_inches="tight")
    print("saved map figure")


if __name__ == "__main__":
    main()
