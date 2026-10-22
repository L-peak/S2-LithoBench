"""Derived statistics used in the manuscript: paired fold comparisons,
CV optimism, transfer baselines and summary tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

FEATURE_SETS = ["S", "SI", "SIT", "SITT"]


def paired_tests(region: str) -> pd.DataFrame:
    res = pd.read_csv(RESULTS / f"metrics_{region}.csv")
    sub = res[(res.protocol == "spatial") & (res.model == "RF")]
    rows = []
    pairs = [("S", "SI"), ("SI", "SIT"), ("SIT", "SITT")]
    for a, b in pairs:
        va = sub[sub.feature_set == a].sort_values("fold")["macro_f1"].to_numpy()
        vb = sub[sub.feature_set == b].sort_values("fold")["macro_f1"].to_numpy()
        t, p = stats.ttest_rel(vb, va)
        rows.append(
            dict(
                region=region,
                comparison=f"{b} - {a}",
                mean_a=va.mean(),
                mean_b=vb.mean(),
                delta=vb.mean() - va.mean(),
                t=t,
                p=p,
                n_folds=len(va),
            )
        )
    return pd.DataFrame(rows)


def optimism(region: str) -> pd.DataFrame:
    res = pd.read_csv(RESULTS / f"metrics_{region}.csv")
    agg = (
        res.groupby(["feature_set", "model", "protocol"])["macro_f1"].mean().unstack("protocol")
    )
    agg["optimism"] = agg["random"] - agg["spatial"]
    return agg.reset_index()


def transfer_baselines() -> pd.DataFrame:
    rows = []
    for train_region, test_region in (
        ("mount_isa", "flinders"),
        ("flinders", "mount_isa"),
    ):
        te = pd.read_csv(DATA / f"features_{test_region}.csv")
        te = te[te["lith_class"].isin(["regolith", "clastic_sedimentary", "carbonate"])]
        majority = te["lith_class"].mode().iloc[0]
        pred = np.full(len(te), majority)
        from sklearn.metrics import accuracy_score, f1_score

        rows.append(
            dict(
                train=train_region,
                test=test_region,
                majority_class=majority,
                oa=accuracy_score(te["lith_class"], pred),
                macro_f1=f1_score(te["lith_class"], pred, average="macro"),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    tests, opts = [], []
    for region in ("mount_isa", "flinders"):
        tests.append(paired_tests(region))
        opts.append(optimism(region).assign(region=region))
    tests = pd.concat(tests, ignore_index=True)
    opts = pd.concat(opts, ignore_index=True)
    tests.to_csv(RESULTS / "paired_tests.csv", index=False)
    opts.to_csv(RESULTS / "cv_optimism.csv", index=False)
    tb = transfer_baselines()
    tm = pd.read_csv(RESULTS / "transfer_metrics.csv")
    tb = tb.merge(tm, on=["train", "test"], suffixes=("_majority", "_rf"))
    tb.to_csv(RESULTS / "transfer_with_baseline.csv", index=False)

    print("== paired tests (spatial CV, RF) ==")
    print(tests.round(3).to_string(index=False))
    print("\n== CV optimism (random - spatial, macro F1) ==")
    piv = opts.pivot_table(index=["region", "feature_set"], columns="model", values="optimism")
    print(piv.round(3).to_string())
    print("\n== transfer: RF vs majority baseline ==")
    print(tb.round(3).to_string(index=False))

    # key numbers for the abstract
    key = {}
    for region in ("mount_isa", "flinders"):
        res = pd.read_csv(RESULTS / f"metrics_{region}.csv")
        s = res[(res.feature_set == "SITT") & (res.model == "RF") & (res.protocol == "spatial")]
        r = res[(res.feature_set == "SITT") & (res.model == "RF") & (res.protocol == "random")]
        sp = res[(res.feature_set == "S") & (res.model == "RF") & (res.protocol == "spatial")]
        key[region] = dict(
            random_macro_f1=float(r["macro_f1"].mean()),
            random_macro_f1_sd=float(r["macro_f1"].std()),
            spatial_macro_f1=float(s["macro_f1"].mean()),
            spatial_macro_f1_sd=float(s["macro_f1"].std()),
            spatial_oa=float(s["oa"].mean()),
            spatial_kappa=float(s["kappa"].mean()),
            spectral_only_spatial_macro_f1=float(sp["macro_f1"].mean()),
        )
    json.dump(key, open(RESULTS / "key_numbers.json", "w"), indent=2)
    print("\n", json.dumps(key, indent=2))


if __name__ == "__main__":
    main()
