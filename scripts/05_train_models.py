"""Model comparison with random vs spatially blocked cross-validation.

Feature sets
  S      : 10 Sentinel-2 bands
  SI     : bands + 6 spectral indices
  SIT    : bands + indices + 7 terrain attributes (Copernicus DEM)
  SITT   : SIT + 3 local texture features

Models: Random Forest, Extra Trees, Histogram Gradient Boosting, RBF-SVM, MLP.
Protocols: (1) stratified random 5-fold, (2) spatial 5-fold (k-means blocks).
Plus a cross-region transfer experiment on the three shared classes.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
SEED = 42
MIN_CLASS_N = 30

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDICES = ["ndvi", "iron_oxide", "ferrous", "clay", "swir_ratio", "brightness"]
TERRAIN = [
    "dem_elev",
    "dem_slope",
    "dem_aspect_sin",
    "dem_aspect_cos",
    "dem_curvature",
    "dem_tpi",
    "dem_roughness",
]
TEXTURE = ["tex_B04", "tex_B08", "tex_B11"]

FEATURE_SETS = {
    "S": BANDS,
    "SI": BANDS + INDICES,
    "SIT": BANDS + INDICES + TERRAIN,
    "SITT": BANDS + INDICES + TERRAIN + TEXTURE,
}

SHARED_CLASSES = ["regolith", "clastic_sedimentary", "carbonate"]


def make_models():
    return {
        "RF": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "HGB": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, random_state=SEED
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=SEED),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                max_iter=600,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=SEED,
            ),
        ),
    }


def metrics(y_true, y_pred) -> dict:
    return dict(
        oa=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        weighted_f1=f1_score(y_true, y_pred, average="weighted"),
        kappa=cohen_kappa_score(y_true, y_pred),
    )


def spatial_folds(xy: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=n_splits, n_init=10, random_state=seed)
    return km.fit_predict(StandardScaler().fit_transform(xy))


def cross_validate(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    protocol: str,
    n_splits: int = 5,
    seed: int = SEED,
) -> list[dict]:
    X = df[features].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if protocol == "random":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(X, y))
    else:
        block = spatial_folds(df[["x_utm", "y_utm"]].to_numpy(), n_splits, seed)
        for k in range(n_splits):
            test = np.where(block == k)[0]
            train = np.where(block != k)[0]
            if len(test) and len(np.unique(y[train])) > 1:
                folds.append((train, test))

    rows = []
    for i, (tr, te) in enumerate(folds):
        model = make_models()[model_name]
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        m = metrics(y[te], pred)
        m.update(fold=i, n_train=len(tr), n_test=len(te))
        rows.append(m)
    return rows


def run_region(region: str) -> None:
    df = pd.read_csv(DATA / f"features_{region}.csv")
    needed = BANDS + INDICES + TERRAIN + TEXTURE
    df = df.dropna(subset=needed + ["lith_class", "x_utm", "y_utm"]).reset_index(drop=True)
    counts = df["lith_class"].value_counts()
    keep = counts[counts >= MIN_CLASS_N].index
    df = df[df["lith_class"].isin(keep)].reset_index(drop=True)
    le = LabelEncoder().fit(sorted(df["lith_class"].unique()))
    df["y"] = le.transform(df["lith_class"])
    print(f"\n=== {region}: {len(df)} samples, {len(le.classes_)} classes ===")
    print(df["lith_class"].value_counts().to_string())

    all_rows = []
    oof_store = {}
    for fs_name, feats in FEATURE_SETS.items():
        for model_name in make_models():
            for protocol in ("random", "spatial"):
                rows = cross_validate(df, feats, model_name, protocol)
                for r in rows:
                    r.update(region=region, feature_set=fs_name, model=model_name, protocol=protocol)
                all_rows.extend(rows)
                m = np.mean([[r["oa"], r["macro_f1"], r["kappa"]] for r in rows], axis=0)
                print(
                    f"  {fs_name:4s} {model_name:3s} {protocol:7s} "
                    f"OA={m[0]:.3f} macroF1={m[1]:.3f} kappa={m[2]:.3f}"
                )

    res = pd.DataFrame(all_rows)
    res.to_csv(RESULTS / f"metrics_{region}.csv", index=False)

    # summary table (mean over folds)
    summary = (
        res.groupby(["feature_set", "model", "protocol"])[["oa", "macro_f1", "kappa"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(RESULTS / f"metrics_summary_{region}.csv")

    # out-of-fold predictions for the best configuration (by spatial macro-F1)
    best = (
        res[res.protocol == "spatial"]
        .groupby(["feature_set", "model"])["macro_f1"]
        .mean()
        .idxmax()
    )
    fs_name, model_name = best
    print(f"  best spatial config: {fs_name} / {model_name}")
    feats = FEATURE_SETS[fs_name]
    X = df[feats].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    block = spatial_folds(df[["x_utm", "y_utm"]].to_numpy(), 5, SEED)
    oof = np.full(len(df), -1)
    for k in range(5):
        tr, te = np.where(block != k)[0], np.where(block == k)[0]
        model = make_models()[model_name]
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    oof_df = df[["region", "lon", "lat", "lith_class", "x_utm", "y_utm"]].copy()
    oof_df["pred_class"] = le.inverse_transform(oof)
    oof_df["correct"] = oof_df["lith_class"] == oof_df["pred_class"]
    oof_df.to_csv(RESULTS / f"oof_predictions_{region}.csv", index=False)

    rep = classification_report(y, oof, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(RESULTS / f"per_class_{region}.csv")
    cm = confusion_matrix(y, oof)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
        RESULTS / f"confusion_{region}.csv"
    )
    json.dump(
        dict(region=region, feature_set=fs_name, model=model_name, classes=list(le.classes_)),
        open(RESULTS / f"best_config_{region}.json", "w"),
        indent=2,
    )

    # permutation importance on a 30% hold-out for the best RF-like model
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = max(300, int(0.3 * len(df)))
    te, tr = idx[:n_test], idx[n_test:]
    model = make_models()[model_name]
    model.fit(X[tr], y[tr])
    imp = permutation_importance(
        model, X[te], y[te], n_repeats=5, random_state=SEED, n_jobs=-1
    )
    imp_df = pd.DataFrame(
        dict(feature=feats, importance=imp.importances_mean, std=imp.importances_std)
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(RESULTS / f"importance_{region}.csv", index=False)

    # final model trained on all samples for the demonstration map
    final_model = make_models()[model_name]
    final_model.fit(X, y)
    joblib.dump(
        dict(model=final_model, features=feats, classes=list(le.classes_), region=region),
        RESULTS / f"model_{region}.joblib",
    )


def run_transfer() -> None:
    frames = {}
    for region in ("mount_isa", "flinders"):
        df = pd.read_csv(DATA / f"features_{region}.csv")
        needed = BANDS + INDICES + TERRAIN + TEXTURE
        df = df.dropna(subset=needed).reset_index(drop=True)
        df = df[df["lith_class"].isin(SHARED_CLASSES)].reset_index(drop=True)
        frames[region] = df
        print(f"transfer {region}: {len(df)} samples of shared classes")

    feats = FEATURE_SETS["SITT"]
    rows = []
    for train_region, test_region in (("mount_isa", "flinders"), ("flinders", "mount_isa")):
        tr_df, te_df = frames[train_region], frames[test_region]
        le = LabelEncoder().fit(sorted(SHARED_CLASSES))
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(tr_df[feats].to_numpy(float), le.transform(tr_df["lith_class"]))
        pred = model.predict(te_df[feats].to_numpy(float))
        truth = le.transform(te_df["lith_class"])
        m = metrics(truth, pred)
        rows.append(dict(train=train_region, test=test_region, **m))
        print(f"  {train_region} -> {test_region}: OA={m['oa']:.3f} macroF1={m['macro_f1']:.3f}")
        cm = confusion_matrix(truth, pred)
        pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
            RESULTS / f"transfer_confusion_{train_region}_to_{test_region}.csv"
        )
    pd.DataFrame(rows).to_csv(RESULTS / "transfer_metrics.csv", index=False)


if __name__ == "__main__":
    for r in ("mount_isa", "flinders"):
        run_region(r)
    run_transfer()
