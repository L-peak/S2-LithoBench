"""Feasibility test: search Sentinel-2 L2A via Earth Search STAC and read a COG window.

No credentials required: assets are public COGs on AWS (us-west-2).
"""
from __future__ import annotations

import json

import rasterio
from pystac_client import Client
from rasterio.windows import Window

STAC_URL = "https://earth-search.aws.element84.com/v1"
# Small test bbox in the Flinders Ranges, South Australia (arid, well exposed).
BBOX = (138.55, -31.45, 138.60, -31.40)
DATETIME = "2024-01-01/2024-03-31"


def main() -> None:
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATETIME,
        query={"eo:cloud_cover": {"lt": 10}},
        max_items=5,
    )
    items = list(search.items())
    print(f"items found: {len(items)}")
    if not items:
        raise SystemExit("no items found")

    item = min(items, key=lambda it: it.properties["eo:cloud_cover"])
    print("item id:", item.id)
    print("datetime:", item.datetime)
    print("cloud cover:", item.properties["eo:cloud_cover"])
    print("assets:", json.dumps(sorted(item.assets.keys()), ensure_ascii=False))

    href = item.assets["red"].href
    print("red href:", href)

    env_opts = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        AWS_NO_SIGN_REQUEST="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    )
    with rasterio.Env(**env_opts):
        with rasterio.open(href) as src:
            print("crs:", src.crs, "size:", src.width, src.height, "dtype:", src.dtypes)
            cx, cy = src.width // 2, src.height // 2
            win = Window(cx - 128, cy - 128, 256, 256)
            arr = src.read(1, window=win)
            print("window shape:", arr.shape, "dtype:", arr.dtype)
            print(
                "window stats: min=%s max=%s mean=%.1f"
                % (arr.min(), arr.max(), float(arr.mean()))
            )
            print("transform:", src.window_transform(win))


if __name__ == "__main__":
    main()
