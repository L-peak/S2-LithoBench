# S2-LithoBench v1.0

**Archived release:** https://doi.org/10.5281/zenodo.23168523

Code and derived data for the manuscript:

> **S2-LithoBench: a reproducible benchmark for validation-aware lithological
> mapping with Sentinel-2 in arid terrains**
> Zhaoyang Li, Henan Polytechnic University

## What this deposit contains

| Folder | Contents |
| --- | --- |
| `scripts/` | Complete pipeline (8 Python scripts, run in order 02-09) |
| `data/` | Sample tables with all Sentinel-2 + terrain features (`features_*.csv`), raw sampled points, and the downloaded geology polygons |
| `results/` | Metric tables, confusion matrices, out-of-fold predictions, permutation importance, paired tests, transfer results, run logs |
| `figures/` | All manuscript figures (PNG 300 dpi and vector PDF) |
| `requirements.txt` | Exact Python package versions used |
| `LICENSE` | MIT licence for the code |
| `LICENSE-DATA` | CC BY 4.0 for the derived data tables |

The trained model files (`*.joblib`, ~100 MB) are not included; they are
regenerated in about 13 minutes by `scripts/05_train_models.py`.

## Robustness battery

| Test | Mount Isa | Flinders Ranges |
| --- | --- | --- |
| Repeated spatial CV, 10 blocking realizations (SITT) | 0.209 | 0.285 |
| Buffered spatial CV (2 km) | 0.191 | 0.272 |
| Block size 20 at fixed 5 folds | 0.207 | 0.256 |
| Geological-unit GroupKFold | 0.091 | 0.109 |
| Label-shuffled / majority-class control | 0.145 / 0.065 | 0.160 / 0.127 |
| Terrain-only features | 0.199 | 0.279 |
| XGBoost, spatial validation | 0.211 | 0.245 |

## How to reproduce

```bash
python -m pip install -r requirements.txt
python scripts/02_query_geology.py       # download GA 1:1M geology polygons
python scripts/03_make_samples.py        # harmonise classes, sample points
python scripts/04_extract_features.py    # Sentinel-2 + DEM feature extraction
python scripts/05_train_models.py        # 4 feature sets x 5 models x 2 protocols
python scripts/06_figures.py             # manuscript figures
python scripts/07_map_demo.py            # 8 km x 8 km demonstration map
python scripts/08_analysis.py            # paired tests, CV optimism, baselines
python scripts/09_robustness.py          # repeated spatial CV, buffers, GroupKFold, controls, XGBoost
```

The pipeline needs only a Python environment and an internet connection; all
source data are public and require no credentials. Measured end-to-end runtime
on a 16-core desktop without GPU: approximately 45 minutes.

## Data sources

* Sentinel-2 L2A COGs via the Earth Search STAC API (Element 84 / AWS Open
  Data), `https://earth-search.aws.element84.com/v1`
* Geoscience Australia *Surface Geology of Australia* 1:1M web map service
  (CC BY 4.0), layer `AUS_GA_1M_GUPoly_Lithology`
* Copernicus DEM GLO-30 COGs on AWS (`copernicus-dem-30m` bucket)

## Key result

| Region | Random CV macro-F1 | Spatial CV macro-F1 | Optimism |
| --- | --- | --- | --- |
| Mount Isa (6 classes, n = 2014) | 0.522 | 0.208 | +0.314 |
| Flinders Ranges (5 classes, n = 1278) | 0.658 | 0.289 | +0.370 |

Terrain attributes were the only feature group with a statistically
significant contribution (paired t-tests: p = 0.012 and p = 0.050). Spectral
indices and 3x3 texture were within noise. Cross-region transfer remained near
chance (Cohen's kappa 0.005-0.029).

## Licence

Code: MIT (`LICENSE`). Derived data tables: CC BY 4.0 (`LICENSE-DATA`).
Source imagery and reference maps remain under their original licences.
