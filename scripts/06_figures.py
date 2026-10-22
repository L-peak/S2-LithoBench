"""Create the manuscript figures from the extracted features and results."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "regolith": "#d9d9d9",
    "clastic_sedimentary": "#f2c14e",
    "carbonate": "#4c9f70",
    "metasedimentary": "#7b6cf6",
    "mafic_igneous": "#d1495b",
    "felsic_intrusive": "#e07a5f",
    "felsic_volcanic": "#2e86ab",
}
CLASS_LABELS = {
    "regolith": "Regolith",
    "clastic_sedimentary": "Clastic sed.",
    "carbonate": "Carbonate",
    "metasedimentary": "Metasedimentary",
    "mafic_igneous": "Mafic igneous",
    "felsic_intrusive": "Felsic intrusive",
    "felsic_volcanic": "Felsic volcanic",
}
REGION_LABELS = {"mount_isa": "Mount Isa (QLD)", "flinders": "Flinders Ranges (SA)"}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def fig_study_areas() -> None:
    from shapely.geometry import box

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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, region in zip(axes, ["mount_isa", "flinders"]):
        bbox = {
            "mount_isa": (139.40, -20.80, 139.70, -20.55),
            "flinders": (138.50, -31.50, 138.80, -31.25),
        }[region]
        geo = gpd.read_file(DATA / f"geology_{region}.geojson").set_crs(4326, allow_override=True)
        geo = geo[geo.geometry.intersects(box(*bbox))].copy()
        geo["cls"] = geo["lithology"].map(harmonise)
        geo = geo[geo["cls"].notna()].copy()
        geo["geometry"] = geo.geometry.intersection(box(*bbox))
        geo = geo[~geo.geometry.is_empty]
        pts = pd.read_csv(DATA / f"features_{region}.csv")
        for cls, grp in geo.groupby("cls"):
            grp.plot(
                ax=ax,
                color=CLASS_COLORS.get(cls, "#cccccc"),
                edgecolor="0.35",
                linewidth=0.15,
                alpha=0.55,
            )
        for cls, grp in pts.groupby("lith_class"):
            ax.scatter(
                grp["lon"],
                grp["lat"],
                s=2.2,
                c=CLASS_COLORS.get(cls, "#333333"),
                label=CLASS_LABELS.get(cls, cls),
                linewidths=0,
            )
        ax.set_title(REGION_LABELS[region])
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlim(bbox[0] - 0.01, bbox[2] + 0.01)
        ax.set_ylim(bbox[1] - 0.01, bbox[3] + 0.01)
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean(bbox[1::2]))))
        ax.grid(linewidth=0.2, alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", frameon=True, markerscale=2.4,
                   handletextpad=0.2, borderpad=0.3)
    save(fig, "fig1_study_areas")


def fig_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, "Sentinel-2 L2A\n(20 m grid, 5 dates)\nSCL cloud mask"),
        (0.27, 0.55, "3×3 neighbourhood\nmedian across dates\n10 bands + 3 textures"),
        (0.52, 0.55, "GA 1:1M surface\ngeology polygons\nharmonised classes"),
        (0.77, 0.55, "Copernicus DEM\nslope / aspect / TPI\ncurvature / roughness"),
        (0.27, 0.08, "Random vs. spatial\n5-fold CV (5 models)"),
        (0.52, 0.08, "Cross-region transfer\n(shared classes)"),
        (0.77, 0.08, "Feature importance\nand class maps"),
    ]
    for x, y, text in boxes:
        ax.add_patch(Rectangle((x, y), 0.21, 0.34, facecolor="#eef3f8",
                               edgecolor="#355070", linewidth=0.8))
        ax.text(x + 0.105, y + 0.17, text, ha="center", va="center", fontsize=7)
    arrows = [
        ((0.23, 0.72), (0.27, 0.72)),
        ((0.48, 0.72), (0.52, 0.72)),
        ((0.73, 0.72), (0.77, 0.72)),
        ((0.37, 0.55), (0.37, 0.42)),
        ((0.62, 0.55), (0.62, 0.42)),
        ((0.87, 0.55), (0.87, 0.42)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, color="#355070", linewidth=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig2_workflow")


def fig_cv_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, region in zip(axes, ["mount_isa", "flinders"]):
        res = pd.read_csv(RESULTS / f"metrics_{region}.csv")
        sub = res[(res.model == "RF")]
        order = ["S", "SI", "SIT", "SITT"]
        width = 0.38
        xpos = np.arange(len(order))
        for i, (proto, color) in enumerate((("random", "#94a3b8"), ("spatial", "#1d4ed8"))):
            means, stds = [], []
            for fs in order:
                vals = sub[(sub.feature_set == fs) & (sub.protocol == proto)]["macro_f1"]
                means.append(vals.mean())
                stds.append(vals.std())
            ax.bar(xpos + (i - 0.5) * width, means, width, yerr=stds, capsize=2,
                   color=color, label=proto.capitalize())
        ax.set_xticks(xpos)
        ax.set_xticklabels(["S", "S+I", "S+I+T", "S+I+T+X"])
        ax.set_title(REGION_LABELS[region])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro F1")
    axes[0].legend(loc="upper left", frameon=False)
    save(fig, "fig3_cv_comparison")


def fig_confusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)
    for ax, region in zip(axes, ["mount_isa", "flinders"]):
        cm = pd.read_csv(RESULTS / f"confusion_{region}.csv", index_col=0)
        norm = cm.to_numpy() / np.maximum(cm.sum(axis=1).to_numpy()[:, None], 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        labels = [CLASS_LABELS.get(c, c) for c in cm.index]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_title(REGION_LABELS[region])
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.043, pad=0.03, shrink=0.85)
    save(fig, "fig4_confusion")


def fig_importance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, region in zip(axes, ["mount_isa", "flinders"]):
        imp = pd.read_csv(RESULTS / f"importance_{region}.csv").head(15).iloc[::-1]
        ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#355070")
        ax.set_title(REGION_LABELS[region])
        ax.set_xlabel("Permutation importance (macro F1 loss)")
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    save(fig, "fig5_importance")


def fig_transfer() -> None:
    tm = pd.read_csv(RESULTS / "transfer_metrics.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    labels = [f"{r.train}\n→ {r.test}" for r in tm.itertuples()]
    x = np.arange(len(tm))
    ax.bar(x - 0.2, tm["oa"], 0.38, label="OA", color="#1d4ed8")
    ax.bar(x + 0.2, tm["macro_f1"], 0.38, label="Macro F1", color="#f2c14e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig6_transfer")


def fig_points_map() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for row, region in enumerate(["mount_isa", "flinders"]):
        oof = pd.read_csv(RESULTS / f"oof_predictions_{region}.csv")
        for col, colname in enumerate(("lith_class", "pred_class")):
            ax = axes[row, col]
            for cls, grp in oof.groupby(colname):
                ax.scatter(grp["lon"], grp["lat"], s=4,
                           c=CLASS_COLORS.get(cls, "#333"), label=CLASS_LABELS.get(cls, cls),
                           linewidths=0)
            ax.set_title(f"{REGION_LABELS[region]} – {'reference' if col == 0 else 'predicted'}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row == 0 and col == 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "fig7_point_maps")


def fig_optimism_heatmap() -> None:
    """Random-minus-spatial macro-F1 for every classifier x feature-set pair."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    feats = ["S", "SI", "SIT", "SITT"]
    models = ["RF", "ET", "HGB", "SVM", "MLP"]
    for ax, region in zip(axes, ["mount_isa", "flinders"]):
        res = pd.read_csv(RESULTS / f"metrics_{region}.csv")
        grid = np.full((len(feats), len(models)), np.nan)
        for i, fs in enumerate(feats):
            for j, m in enumerate(models):
                a = res[(res.feature_set == fs) & (res.model == m)
                        & (res.protocol == "random")]["macro_f1"].mean()
                b = res[(res.feature_set == fs) & (res.model == m)
                        & (res.protocol == "spatial")]["macro_f1"].mean()
                grid[i, j] = a - b
        im = ax.imshow(grid, cmap="magma", vmin=0, vmax=0.45)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(["S", "S+I", "S+I+T", "S+I+T+X"])
        for i in range(len(feats)):
            for j in range(len(models)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if grid[i, j] < 0.28 else "black")
        ax.set_title(REGION_LABELS[region])
        ax.set_xlabel("Classifier")
    axes[0].set_ylabel("Feature set")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                 label="Random − spatial macro-F1")
    save(fig, "fig9_optimism_heatmap")


def fig_validation_gradient() -> None:
    """Macro-F1 as validation becomes progressively stricter."""
    stages = ["Random CV", "Spatial CV", "Buffered\nspatial CV", "Geological-unit\nGroupKFold"]
    mi = [0.522, 0.209, 0.191, 0.091]
    fl = [0.658, 0.285, 0.272, 0.109]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.axhspan(0.145, 0.160, color="#c7c7c7", alpha=0.45, zorder=0,
               label="Label-shuffled control")
    ax.axhline(0.065, color="#888888", linestyle=":", linewidth=1,
               label="Majority-class control (Mount Isa)")
    ax.axhline(0.127, color="#bbbbbb", linestyle=":", linewidth=1,
               label="Majority-class control (Flinders)")
    ax.plot(x, mi, "-o", color="#1d4ed8", linewidth=1.4, markersize=5,
            label="Mount Isa")
    ax.plot(x, fl, "-s", color="#d1495b", linewidth=1.4, markersize=5,
            label="Flinders Ranges")
    for xi, (a, b) in enumerate(zip(mi, fl)):
        ax.annotate(f"{a:.3f}", (xi, a), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5, color="#1d4ed8")
        ax.annotate(f"{b:.3f}", (xi, b), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.5, color="#d1495b")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylabel("Macro-$F_1$")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=6.2)
    save(fig, "fig10_validation_gradient")


if __name__ == "__main__":
    fig_workflow()
    fig_study_areas()
    fig_cv_comparison()
    fig_confusion()
    fig_importance()
    fig_transfer()
    fig_points_map()
    fig_optimism_heatmap()
    fig_validation_gradient()
