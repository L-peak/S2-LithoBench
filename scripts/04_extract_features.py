"""Extract Sentinel-2 spectral features and Copernicus DEM terrain features
at the labelled sample points.

Design notes
------------
* The 1:1M geological labels do not justify 10 m pixels, so the working grid
  is the native 20 m Sentinel-2 grid.  This cuts download volume by ~4x and
  keeps the 3x3 windows (60 m) well inside the sampled map polygons.
* Memory is limited (~16 GB, mostly busy), so each band/date is read once,
  reduced to 3x3 summary statistics at the sample points, and released.
* Bands within a date are read in parallel with a small thread pool.
* Cloud screening uses the 20 m Scene Classification Layer (SCL); a date
  contributes for a sample when at least 7 of the 9 surrounding pixels are
  bare ground, vegetation, or unclassified.

Public data, no credentials:
* Sentinel-2 L2A COGs, Earth Search STAC (Element 84 / AWS).
* Copernicus DEM GLO-30 COGs on AWS.
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds, intersection

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://earth-search.aws.element84.com/v1"
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
WORK_RES = 20.0  # working grid resolution in metres

REGIONS = {
    "mount_isa": dict(
        bbox=(139.40, -20.80, 139.70, -20.55),
        datetime="2024-04-01/2024-10-31",
    ),
    "flinders": dict(
        bbox=(138.50, -31.50, 138.80, -31.25),
        datetime="2024-04-01/2024-10-31",
    ),
}

BANDS_10M = [("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir", "B08")]
BANDS_20M = [
    ("rededge1", "B05"),
    ("rededge2", "B06"),
    ("rededge3", "B07"),
    ("nir08", "B8A"),
    ("swir16", "B11"),
    ("swir22", "B12"),
]
ALL_BANDS = [n for _, n in BANDS_10M + BANDS_20M]
VALID_SCL = np.array([4, 5, 7])
MIN_VALID_DATES = 3
MIN_SCL_PIXELS = 7
MAX_DATES = 5
OFFSET_KEY = "earthsearch:boa_offset_applied"

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    CPL_VSIL_CURL_CHUNK_SIZE="1048576",
    AWS_NO_SIGN_REQUEST="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="200000000",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reference_grid(item, bounds):
    """Window/transform/shape of the working grid, aligned to the 20 m SCL grid."""
    href = item.assets["scl"].href
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            win = from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            win = intersection(win, Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None
            shape = (int(round(win.height)), int(round(win.width)))
            snapped = window_bounds(win, src.transform)
            transform = src.window_transform(win)
    return snapped, transform, shape


def read_band_window(asset_key: str, item, snapped, shape) -> np.ndarray | None:
    href = item.assets[asset_key].href
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            win = from_bounds(*snapped, transform=src.transform)
            win = win.round_offsets().round_lengths()
            win = intersection(win, Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None
            return src.read(1, window=win, out_shape=shape, resampling=Resampling.bilinear)


def patch_mean_std(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    n = len(rows)
    mean = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    h, w = arr.shape
    for i in range(n):
        r, c = rows[i], cols[i]
        if r < 1 or c < 1 or r >= h - 1 or c >= w - 1:
            continue
        p = arr[r - 1 : r + 2, c - 1 : c + 2].astype(np.float32)
        if not np.isfinite(p).all():
            continue
        mean[i] = p.mean()
        std[i] = p.std()
    return mean, std


def scl_valid_count(scl: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    n = len(rows)
    cnt = np.zeros(n, dtype=np.int16)
    h, w = scl.shape
    for i in range(n):
        r, c = rows[i], cols[i]
        if r < 1 or c < 1 or r >= h - 1 or c >= w - 1:
            continue
        p = scl[r - 1 : r + 2, c - 1 : c + 2]
        cnt[i] = np.isin(p, VALID_SCL).sum()
    return cnt


def select_items(catalog: Client, bbox, datetime: str, max_items: int):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=80,
    )
    items = list(search.items())
    covering = [
        it
        for it in items
        if it.bbox[0] <= bbox[0]
        and it.bbox[1] <= bbox[1]
        and it.bbox[2] >= bbox[2]
        and it.bbox[3] >= bbox[3]
    ]
    covering.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    chosen = covering[:max_items]
    log(f"  STAC candidates: {len(items)}, fully covering: {len(covering)}")
    log(
        "  chosen: "
        + ", ".join(
            f"{it.datetime:%Y-%m-%d}({it.properties.get('eo:cloud_cover', -1):.1f}%)"
            for it in chosen
        )
    )
    return chosen


def extract_region(region: str, max_dates: int, band_limit: int | None) -> None:
    cfg = REGIONS[region]
    samples = pd.read_csv(DATA / f"samples_{region}.csv")
    log(f"{region}: {len(samples)} samples")

    transformer = Transformer.from_crs(4326, 32754, always_xy=True)
    xs, ys = transformer.transform(samples["lon"].values, samples["lat"].values)
    margin = 300.0
    bounds = (xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin)

    catalog = Client.open(STAC_URL)
    items = select_items(catalog, cfg["bbox"], cfg["datetime"], max_dates)
    if not items:
        raise SystemExit(f"{region}: no Sentinel-2 items")

    n_pts = len(samples)
    band_mean = {b: np.full((n_pts, len(items)), np.nan, dtype=np.float32) for b in ALL_BANDS}
    band_std = {b: np.full((n_pts, len(items)), np.nan, dtype=np.float32) for b in ALL_BANDS}
    valid_dates = np.zeros((n_pts, len(items)), dtype=bool)

    pairs = BANDS_10M + BANDS_20M
    if band_limit:
        pairs = pairs[:band_limit]

    for di, item in enumerate(items):
        offset = 0.0 if item.properties.get(OFFSET_KEY, True) else 1000.0
        t0 = time.time()
        grid = reference_grid(item, bounds)
        if grid is None:
            log(f"  {item.id}: window outside tile, skipped")
            continue
        snapped, transform, shape = grid
        rows = np.round((xs - transform.c) / transform.a - 0.5).astype(int)
        cols = np.round((ys - transform.f) / transform.e - 0.5).astype(int)

        scl = read_band_window("scl", item, snapped, shape)
        if scl is None:
            continue
        n_ok = scl_valid_count(scl, rows, cols)
        usable = n_ok >= MIN_SCL_PIXELS
        valid_dates[:, di] = usable
        del scl
        log(
            f"  {item.id} ({item.datetime:%Y-%m-%d}) grid={shape} "
            f"usable={usable.sum()}/{n_pts} scl_done={time.time() - t0:.0f}s"
        )

        def worker(asset_key: str, out_name: str):
            arr = read_band_window(asset_key, item, snapped, shape)
            if arr is None:
                return out_name, None, None
            if offset:
                arr = (arr - offset) / 10000.0
            else:
                arr = arr / 10000.0
            m, s = patch_mean_std(arr, rows, cols)
            m[~usable] = np.nan
            s[~usable] = np.nan
            return out_name, m, s

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, k, n) for k, n in pairs]
            for fut in as_completed(futures):
                out_name, m, s = fut.result()
                if m is not None:
                    band_mean[out_name][:, di] = m
                    band_std[out_name][:, di] = s
        log(f"  {item.id}: bands done in {time.time() - t0:.0f}s")

    out = samples[
        [
            "region",
            "lon",
            "lat",
            "x_utm",
            "y_utm",
            "lith_class",
            "lithology_raw",
            "unit_name",
            "polygon_area_km2",
        ]
    ].copy()
    for b in ALL_BANDS:
        out[b] = np.nanmedian(band_mean[b], axis=1)
        out[f"tex_{b}"] = np.nanmedian(band_std[b], axis=1)
    out["n_valid_dates"] = valid_dates.sum(axis=1)
    log(f"  valid-date histogram: {np.bincount(out['n_valid_dates'].values, minlength=len(items) + 1).tolist()}")
    out = out[out["n_valid_dates"] >= MIN_VALID_DATES].reset_index(drop=True)
    log(f"  kept after >= {MIN_VALID_DATES} dates: {len(out)}/{n_pts}")

    eps = 1e-6
    out["ndvi"] = (out["B08"] - out["B04"]) / (out["B08"] + out["B04"] + eps)
    out["iron_oxide"] = out["B04"] / (out["B02"] + eps)
    out["ferrous"] = out["B11"] / (out["B08"] + eps)
    out["clay"] = out["B11"] / (out["B12"] + eps)
    out["swir_ratio"] = (out["B11"] - out["B12"]) / (out["B11"] + out["B12"] + eps)
    out["brightness"] = out[["B02", "B03", "B04"]].mean(axis=1)
    out.to_csv(DATA / f"features_s2_{region}.csv", index=False)
    log(f"  saved {DATA / f'features_s2_{region}.csv'}")


def dem_tile_name(lat: int, lon: int) -> str:
    ns = "S" if lat < 0 else "N"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def extract_dem(region: str) -> None:
    cfg = REGIONS[region]
    src_csv = DATA / f"features_s2_{region}.csv"
    samples = pd.read_csv(src_csv)
    # The DEM mosaic is in EPSG:4326, so index it directly with lon/lat.
    xs = samples["lon"].to_numpy()
    ys = samples["lat"].to_numpy()

    lon0, lat0, lon1, lat1 = cfg["bbox"]
    tiles = set()
    for lat in (math.floor(lat0), math.floor(lat1)):
        for lon in (math.floor(lon0), math.floor(lon1)):
            name = dem_tile_name(lat, lon)
            tiles.add(f"{DEM_BUCKET}/{name}/{name}.tif")

    from rasterio.merge import merge
    from scipy.ndimage import uniform_filter

    with rasterio.Env(**GDAL_ENV):
        srcs = []
        for href in sorted(tiles):
            try:
                srcs.append(rasterio.open(href))
            except Exception as exc:  # noqa: BLE001
                log(f"  DEM unavailable: {href} ({exc})")
        if not srcs:
            raise SystemExit(f"{region}: no DEM tiles")
        mosaic, transform = merge(srcs)
        for src in srcs:
            src.close()
    dem = mosaic[0].astype(np.float32)
    del mosaic

    # The DEM mosaic is a geographic (degree) grid: convert gradients to metres.
    lat_mean = float(samples["lat"].mean())
    res_deg = abs(transform.a)
    dx_m = res_deg * 111320.0 * math.cos(math.radians(lat_mean))
    dy_m = res_deg * 110540.0
    d_drow = np.gradient(dem, axis=0)  # change per row (southward)
    d_dcol = np.gradient(dem, axis=1)  # change per column (eastward)
    dzdy = -d_drow / dy_m  # northward derivative
    dzdx = d_dcol / dx_m  # eastward derivative
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # compass bearing of the downslope direction (clockwise from north)
    aspect = np.arctan2(-dzdx, -dzdy)
    d2x = np.gradient(dzdx, axis=1) / dx_m
    d2y = -np.gradient(dzdy, axis=0) / dy_m
    curvature = d2x + d2y
    mean15 = uniform_filter(dem, size=15, mode="nearest")
    tpi = dem - mean15
    m5 = uniform_filter(dem, size=5, mode="nearest")
    m5sq = uniform_filter(dem**2, size=5, mode="nearest")
    roughness = np.sqrt(np.maximum(m5sq - m5**2, 0))

    cols = np.round((xs - transform.c) / transform.a - 0.5).astype(int)
    rows = np.round((ys - transform.f) / transform.e - 0.5).astype(int)
    ok = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    n = len(samples)
    fields = {
        "dem_elev": np.full(n, np.nan),
        "dem_slope": np.full(n, np.nan),
        "dem_aspect_sin": np.full(n, np.nan),
        "dem_aspect_cos": np.full(n, np.nan),
        "dem_curvature": np.full(n, np.nan),
        "dem_tpi": np.full(n, np.nan),
        "dem_roughness": np.full(n, np.nan),
    }
    r, c = rows[ok], cols[ok]
    fields["dem_elev"][ok] = dem[r, c]
    fields["dem_slope"][ok] = slope[r, c]
    fields["dem_aspect_sin"][ok] = np.sin(aspect[r, c])
    fields["dem_aspect_cos"][ok] = np.cos(aspect[r, c])
    fields["dem_curvature"][ok] = curvature[r, c]
    fields["dem_tpi"][ok] = tpi[r, c]
    fields["dem_roughness"][ok] = roughness[r, c]

    merged = pd.concat([samples.reset_index(drop=True), pd.DataFrame(fields)], axis=1)
    merged.to_csv(DATA / f"features_{region}.csv", index=False)
    log(f"  DEM merged -> {DATA / f'features_{region}.csv'} {merged.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both")
    ap.add_argument("--max-dates", type=int, default=MAX_DATES)
    ap.add_argument("--band-limit", type=int, default=None)
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--skip-dem", action="store_true")
    args = ap.parse_args()

    regions = list(REGIONS) if args.region == "both" else [args.region]
    for region in regions:
        if not args.skip_s2:
            log(f"=== Sentinel-2: {region} ===")
            extract_region(region, args.max_dates, args.band_limit)
        if not args.skip_dem:
            log(f"=== DEM: {region} ===")
            extract_dem(region)


if __name__ == "__main__":
    main()
