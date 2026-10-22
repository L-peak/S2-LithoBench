"""Download surface-geology polygons (1:1M) from the Geoscience Australia
ArcGIS MapServer for the two study regions.

Source: GA Surface Geology of Australia, layer 11 (AUS_GA_1M_GUPoly_Lithology).
Licence: CC BY 4.0 (Commonwealth of Australia, Geoscience Australia).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

SERVICE = (
    "https://services.ga.gov.au/gis/rest/services/GA_Surface_Geology/MapServer/11/query"
)

REGIONS = {
    "mount_isa": dict(bbox=(139.40, -20.80, 139.70, -20.55)),
    "flinders": dict(bbox=(138.50, -31.50, 138.80, -31.25)),
}

OUT_FIELDS = "lithology,name,descr,typename,resscale,plotrank,featureid"


def query_geology(bbox: tuple[float, float, float, float]) -> dict:
    params = {
        "where": "1=1",
        "geometry": ",".join(f"{v}" for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": OUT_FIELDS,
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": "2000",
        "f": "geojson",
    }
    url = SERVICE + "?" + urllib.parse.urlencode(params, safe=",")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    for name, meta in REGIONS.items():
        for attempt in range(3):
            try:
                gj = query_geology(meta["bbox"])
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        else:
            raise SystemExit(f"{name}: could not download geology")

        out = DATA / f"geology_{name}.geojson"
        out.write_text(json.dumps(gj), encoding="utf-8")
        feats = gj.get("features", [])
        print(f"{name}: {len(feats)} features -> {out}")


if __name__ == "__main__":
    main()
