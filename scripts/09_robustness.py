"""Robustness experiments requested by the simulated C&G review.

Adds, on top of the main benchmark:
  * repeated spatial CV (10 k-means seeds) with bootstrap 95% CIs,
  * Wilcoxon signed-rank tests with Holm correction,
  * buffered spatial CV (training points within 0.5/1/2 km of test points removed),
  * block-size sensitivity (5, 10, 20 blocks),
  * geological-unit (polygon) GroupKFold,
  * class-balanced and polygon-balanced sampling sensitivity,
  * label-shuffled control,
  * XGBoost baseline,
  * transfer bootstrap CIs and a shuffled-label transfer control.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import bootstrap, wilcoxon
from shapely.geometry import box
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDICES = ["ndvi", "iron_oxide", "ferrous", "clay", "swir_ratio", "brightness"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos",
           "dem_curvature", "dem_tpi", "dem_roughness"]
TEXTURE = ["tex_B04", "tex_B08", "tex_B11"]
FEATURE_SETS = {
    "S": BANDS,
    "SI": BANDS + INDICES,
    "SIT": BANDS + INDICES + TERRAIN,
    "SITT": BANDS + INDICES + TERRAIN + TEXTURE,
    "T": TERRAIN,
    "ST": BANDS + TERRAIN,
}
SHARED = ["regolith", "clastic_sedimentary", "carbonate"]
BBOX = {
    "mount_isa": (139.40, -20.80, 139.70, -20.55),
    "flinders": (138.50, -31.50, 138.80, -31.25),
}
MIN_CLASS_N = 30


def log(msg: str) -> None:
    print(msg, flush=True)


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED,
    )


def xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED,
        eval_metric="mlogloss",
    )


def stats(y, pred) -> dict:
    return dict(
        oa=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
        bal_acc=balanced_accuracy_score(y, pred),
        kappa=cohen_kappa_score(y, pred),
    )


def attach_polygon_ids(region: str, df: pd.DataFrame) -> pd.DataFrame:
    geo = gpd.read_file(DATA / f"geology_{region}.geojson").set_crs(4326, allow_override=True)
    geo = geo[geo.geometry.intersects(box(*BBOX[region]))].copy()
    geo["geometry"] = geo.geometry.intersection(box(*BBOX[region]))
    geo = geo[~geo.geometry.is_empty]
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326
    )
    joined = gpd.sjoin(pts, geo[["featureid", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    missing = joined["featureid"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(pts[missing.values], geo[["featureid", "geometry"]])
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[missing.values, "featureid"] = nearest["featureid"].values
    df = df.copy()
    df["polygon_id"] = joined["featureid"].astype(str).values
    return df


def load_region(region: str):
    df = pd.read_csv(DATA / f"features_{region}.csv")
    need = sorted(set(BANDS + INDICES + TERRAIN + TEXTURE))
    df = df.dropna(subset=need).reset_index(drop=True)
    df = attach_polygon_ids(region, df)
    counts = df["lith_class"].value_counts()
    keep = counts[counts >= MIN_CLASS_N].index
    df = df[df["lith_class"].isin(keep)].reset_index(drop=True)
    le = LabelEncoder().fit(sorted(df["lith_class"].unique()))
    df["y"] = le.transform(df["lith_class"])
    return df, le


def spatial_folds(xy: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(StandardScaler().fit_transform(xy))


def evaluate_folds(df, feats, folds, model_fn, buffer_m=None):
    X = df[feats].to_numpy(float)
    y = df["y"].to_numpy()
    coords = df[["x_utm", "y_utm"]].to_numpy(float)
    rows = []
    for tr, te in folds:
        if buffer_m:
            tree = cKDTree(coords[te])
            d, _ = tree.query(coords[tr])
            tr = tr[d >= buffer_m]
        model = model_fn()
        model.fit(X[tr], y[tr])
        rows.append(stats(y[te], model.predict(X[te])))
    return rows


def evaluate_folds_xgb(df, feats, folds):
    """XGBoost needs contiguous class ids inside every training fold."""
    X = df[feats].to_numpy(float)
    y = df["y"].to_numpy()
    classes_all = np.unique(y)
    rows = []
    for tr, te in folds:
        le = LabelEncoder().fit(y[tr])
        model = xgb()
        model.fit(X[tr], le.transform(y[tr]))
        pred = le.inverse_transform(model.predict(X[te]))
        rows.append(dict(
            oa=accuracy_score(y[te], pred),
            macro_f1=f1_score(y[te], pred, average="macro", labels=classes_all),
            bal_acc=balanced_accuracy_score(y[te], pred),
            kappa=cohen_kappa_score(y[te], pred, labels=classes_all),
        ))
    return rows


def block_folds(xy: np.ndarray, n_blocks: int, n_folds: int = 5, seed: int = 0):
    """Split the area into n_blocks spatial blocks, then merge blocks into
    n_folds folds, so that the number of folds stays fixed while the spatial
    granularity of the blocks changes."""
    b = spatial_folds(xy, n_blocks, seed)
    cent = np.array([xy[b == j].mean(axis=0) for j in range(n_blocks)])
    assign = KMeans(n_clusters=n_folds, n_init=10, random_state=seed).fit_predict(
        StandardScaler().fit_transform(cent)
    )
    folds = []
    for f in range(n_folds):
        test_idx = np.where(np.isin(b, np.where(assign == f)[0]))[0]
        train_idx = np.where(np.isin(b, np.where(assign != f)[0]))[0]
        folds.append((train_idx, test_idx))
    return folds


def repeated_spatial(df, feats, repeats=10, k=5):
    xy = df[["x_utm", "y_utm"]].to_numpy(float)
    out = []
    for r in range(repeats):
        block = spatial_folds(xy, k, seed=r)
        folds = [(np.where(block != j)[0], np.where(block == j)[0]) for j in range(k)]
        rows = evaluate_folds(df, feats, folds, rf)
        for j, m in enumerate(rows):
            out.append(dict(repeat=r, fold=j, **m))
    return pd.DataFrame(out)


def ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    res = bootstrap((values,), np.mean, n_resamples=2000, method="percentile",
                    random_state=SEED)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        val = max(val, prev)
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj.tolist()


def main() -> None:
    main_feats = ["S", "SI", "SIT", "SITT"]
    all_rep, all_sum, tests = [], [], []

    for region in ("mount_isa", "flinders"):
        df, le = load_region(region)
        log(f"=== {region}: {len(df)} samples, {len(le.classes_)} classes, "
            f"{df['polygon_id'].nunique()} geological units ===")

        # --- repeated spatial CV (10 seeds) for the main feature sets
        rep = {}
        for fs in main_feats:
            r = repeated_spatial(df, FEATURE_SETS[fs], repeats=10, k=5)
            r["feature_set"], r["region"] = fs, region
            all_rep.append(r)
            rep[fs] = r.groupby("repeat")["macro_f1"].mean().to_numpy()
            lo, hi = ci95(r.groupby("repeat")["macro_f1"].mean().to_numpy())
            row = dict(
                region=region, feature_set=fs, protocol="spatial_repeated",
                macro_f1_mean=r["macro_f1"].mean(), macro_f1_sd=r["macro_f1"].std(),
                ci_low=lo, ci_high=hi, oa=r["oa"].mean(),
                bal_acc=r["bal_acc"].mean(), kappa=r["kappa"].mean(),
            )
            all_sum.append(row)
            log(f"  {fs:4s} repeated spatial: macro-F1={row['macro_f1_mean']:.3f} "
                f"[{lo:.3f}, {hi:.3f}]")

        # --- Wilcoxon + Holm for the three nested comparisons
        ps, labels = [], []
        for a, b in (("S", "SI"), ("SI", "SIT"), ("SIT", "SITT")):
            stat, p = wilcoxon(rep[b], rep[a])
            ps.append(float(p))
            labels.append(f"{b}-{a}")
            log(f"  Wilcoxon {b} vs {a}: p={p:.4f}")
        adj = holm(ps)
        for lab, p, pa in zip(labels, ps, adj):
            tests.append(dict(region=region, comparison=lab, p=p, p_holm=pa))

        # --- terrain-only and bands+terrain diagnostics (3 repeats)
        for fs in ("T", "ST"):
            r = repeated_spatial(df, FEATURE_SETS[fs], repeats=3, k=5)
            all_sum.append(dict(
                region=region, feature_set=fs, protocol="spatial_repeated",
                macro_f1_mean=r["macro_f1"].mean(), macro_f1_sd=r["macro_f1"].std(),
                ci_low=np.nan, ci_high=np.nan, oa=r["oa"].mean(),
                bal_acc=r["bal_acc"].mean(), kappa=r["kappa"].mean(),
            ))
            log(f"  {fs:4s} repeated spatial (3 seeds): macro-F1={r['macro_f1'].mean():.3f}")

        # --- buffered spatial CV
        xy = df[["x_utm", "y_utm"]].to_numpy(float)
        block = spatial_folds(xy, 5, SEED)
        base_folds = [(np.where(block != j)[0], np.where(block == j)[0]) for j in range(5)]
        for buf in (0, 500, 1000, 2000):
            rows = evaluate_folds(df, FEATURE_SETS["SITT"], base_folds, rf,
                                  buffer_m=buf or None)
            m = np.mean([r["macro_f1"] for r in rows])
            all_sum.append(dict(
                region=region, feature_set="SITT", protocol=f"spatial_buffer_{buf}m",
                macro_f1_mean=m, macro_f1_sd=np.std([r["macro_f1"] for r in rows]),
                ci_low=np.nan, ci_high=np.nan,
                oa=np.mean([r["oa"] for r in rows]),
                bal_acc=np.mean([r["bal_acc"] for r in rows]),
                kappa=np.mean([r["kappa"] for r in rows]),
            ))
            log(f"  buffer {buf:4d} m: macro-F1={m:.3f}")

        # --- block-size sensitivity (number of folds fixed at 5)
        for k in (5, 10, 20):
            vals = []
            for s in range(3):
                folds = block_folds(xy, k, n_folds=5, seed=s)
                rows = evaluate_folds(df, FEATURE_SETS["SITT"], folds, rf)
                vals.append(np.mean([r["macro_f1"] for r in rows]))
            all_sum.append(dict(
                region=region, feature_set="SITT", protocol=f"blocks_{k}",
                macro_f1_mean=float(np.mean(vals)), macro_f1_sd=float(np.std(vals)),
                ci_low=np.nan, ci_high=np.nan, oa=np.nan, bal_acc=np.nan, kappa=np.nan,
            ))
            log(f"  blocks k={k:2d}: macro-F1={np.mean(vals):.3f}")

        # --- geological-unit GroupKFold
        groups = df["polygon_id"].to_numpy()
        gkf = GroupKFold(n_splits=5)
        for fs in main_feats:
            folds = list(gkf.split(df[FEATURE_SETS[fs]], df["y"], groups))
            rows = evaluate_folds(df, FEATURE_SETS[fs], folds, rf)
            all_sum.append(dict(
                region=region, feature_set=fs, protocol="group_polygon_cv",
                macro_f1_mean=np.mean([r["macro_f1"] for r in rows]),
                macro_f1_sd=np.std([r["macro_f1"] for r in rows]),
                ci_low=np.nan, ci_high=np.nan,
                oa=np.mean([r["oa"] for r in rows]),
                bal_acc=np.mean([r["bal_acc"] for r in rows]),
                kappa=np.mean([r["kappa"] for r in rows]),
            ))
            log(f"  polygon GroupKFold {fs}: "
                f"macro-F1={np.mean([r['macro_f1'] for r in rows]):.3f}")

        # --- class-balanced and polygon-balanced sampling
        min_n = int(df["y"].value_counts().min())
        sub = pd.concat(
            [g.sample(min_n, random_state=SEED) for _, g in df.groupby("y")]
        ).reset_index(drop=True)
        r = repeated_spatial(sub, FEATURE_SETS["SITT"], repeats=3, k=5)
        all_sum.append(dict(
            region=region, feature_set="SITT", protocol="class_balanced_sampling",
            macro_f1_mean=r["macro_f1"].mean(), macro_f1_sd=r["macro_f1"].std(),
            ci_low=np.nan, ci_high=np.nan, oa=r["oa"].mean(),
            bal_acc=r["bal_acc"].mean(), kappa=r["kappa"].mean(),
        ))
        log(f"  class-balanced (n/class={min_n}): macro-F1={r['macro_f1'].mean():.3f}")

        cap = df.groupby("polygon_id").size().median()
        cap = int(max(3, min(20, cap)))
        poly_sub = pd.concat(
            [g.sample(min(len(g), cap), random_state=SEED)
             for _, g in df.groupby("polygon_id")]
        ).reset_index(drop=True)
        r = repeated_spatial(poly_sub, FEATURE_SETS["SITT"], repeats=3, k=5)
        all_sum.append(dict(
            region=region, feature_set="SITT", protocol=f"polygon_capped_{cap}",
            macro_f1_mean=r["macro_f1"].mean(), macro_f1_sd=r["macro_f1"].std(),
            ci_low=np.nan, ci_high=np.nan, oa=r["oa"].mean(),
            bal_acc=r["bal_acc"].mean(), kappa=r["kappa"].mean(),
        ))
        log(f"  polygon-capped (<= {cap}/polygon): macro-F1={r['macro_f1'].mean():.3f}")

        # --- label-shuffled control
        vals = []
        for s in range(3):
            rng = np.random.default_rng(s)
            shuffled = df.copy()
            shuffled["y"] = rng.permutation(df["y"].to_numpy())
            b = spatial_folds(xy, 5, s)
            folds = [(np.where(b != j)[0], np.where(b == j)[0]) for j in range(5)]
            rows = evaluate_folds(shuffled, FEATURE_SETS["SITT"], folds, rf)
            vals.append(np.mean([x["macro_f1"] for x in rows]))
        all_sum.append(dict(
            region=region, feature_set="SITT", protocol="label_shuffled",
            macro_f1_mean=float(np.mean(vals)), macro_f1_sd=float(np.std(vals)),
            ci_low=np.nan, ci_high=np.nan, oa=np.nan, bal_acc=np.nan, kappa=np.nan,
        ))
        log(f"  label-shuffled control: macro-F1={np.mean(vals):.3f}")

        # --- XGBoost baseline
        X = df[FEATURE_SETS["SITT"]].to_numpy(float)
        y = df["y"].to_numpy()
        skf_rows = []
        for s in range(3):
            b = spatial_folds(xy, 5, s)
            folds = [(np.where(b != j)[0], np.where(b == j)[0]) for j in range(5)]
            skf_rows += evaluate_folds_xgb(df, FEATURE_SETS["SITT"], folds)
        all_sum.append(dict(
            region=region, feature_set="SITT", protocol="xgb_spatial",
            macro_f1_mean=np.mean([r["macro_f1"] for r in skf_rows]),
            macro_f1_sd=np.std([r["macro_f1"] for r in skf_rows]),
            ci_low=np.nan, ci_high=np.nan,
            oa=np.mean([r["oa"] for r in skf_rows]),
            bal_acc=np.mean([r["bal_acc"] for r in skf_rows]),
            kappa=np.mean([r["kappa"] for r in skf_rows]),
        ))
        log(f"  XGBoost spatial (3 seeds): "
            f"macro-F1={np.mean([r['macro_f1'] for r in skf_rows]):.3f}")

        # incremental checkpoint so a later failure cannot lose finished regions
        pd.concat(all_rep, ignore_index=True).to_csv(
            RESULTS / "robustness_repeated_cv.csv", index=False)
        pd.DataFrame(all_sum).to_csv(RESULTS / "robustness_summary.csv", index=False)
        pd.DataFrame(tests).to_csv(RESULTS / "robustness_wilcoxon.csv", index=False)
        log(f"  checkpoint saved for {region}")

    pd.concat(all_rep, ignore_index=True).to_csv(RESULTS / "robustness_repeated_cv.csv", index=False)
    pd.DataFrame(all_sum).to_csv(RESULTS / "robustness_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "robustness_wilcoxon.csv", index=False)
    log("saved robustness tables")

    # ---------------- transfer robustness ----------------
    frames = {}
    for region in ("mount_isa", "flinders"):
        df, le = load_region(region)
        df = df[df["lith_class"].isin(SHARED)].reset_index(drop=True)
        frames[region] = (df, LabelEncoder().fit(sorted(SHARED)))
    rows = []
    for tr_name, te_name in (("mount_isa", "flinders"), ("flinders", "mount_isa")):
        tr_df, le = frames[tr_name]
        te_df, _ = frames[te_name]
        X_tr = tr_df[FEATURE_SETS["SITT"]].to_numpy(float)
        y_tr = le.transform(tr_df["lith_class"])
        X_te = te_df[FEATURE_SETS["SITT"]].to_numpy(float)
        y_te = le.transform(te_df["lith_class"])
        model = rf()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        base = stats(y_te, pred)
        res = bootstrap((y_te, pred), lambda a, b: f1_score(a, b, average="macro"),
                        n_resamples=1000, method="percentile", paired=True, random_state=SEED)
        shuffle_scores = []
        for s in range(5):
            rng = np.random.default_rng(s)
            m = rf()
            m.fit(X_tr, rng.permutation(y_tr))
            shuffle_scores.append(f1_score(y_te, m.predict(X_te), average="macro"))
        rows.append(dict(
            train=tr_name, test=te_name, **base,
            macro_f1_ci_low=float(res.confidence_interval.low),
            macro_f1_ci_high=float(res.confidence_interval.high),
            shuffled_macro_f1_mean=float(np.mean(shuffle_scores)),
            shuffled_macro_f1_sd=float(np.std(shuffle_scores)),
        ))
        log(f"transfer {tr_name}->{te_name}: macro-F1={base['macro_f1']:.3f} "
            f"[{res.confidence_interval.low:.3f},{res.confidence_interval.high:.3f}], "
            f"shuffled={np.mean(shuffle_scores):.3f}")
    pd.DataFrame(rows).to_csv(RESULTS / "robustness_transfer.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
