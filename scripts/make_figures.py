"""
make_figures.py: draw Figures 2, 3 and 4 of the accompanying manuscript from the
deposited result files.

Each figure is drawn from values already in this repository, so nothing is
refitted, no model binary is loaded and no dataset has to be rebuilt:

  Figure 2  Results/metrics/cv_predictions/cv_predictions_rf1.csv
            out-of-fold predictions of the lumped RF model from 5-fold
            cross-validation, cubic hosts highlighted, with the cubic-only
            MSE, MAE and R2 of each lattice parameter in its panel.
  Figure 3  Results/benchmarks/UNUC/predictions.csv and
            Results/benchmarks/CeO2Nd2O3/predictions.csv
            the five models against the measured lattice parameter of both
            validation systems, side by side.
  Figure 4  Results/metrics/feature_spearman_cubic.csv
            the features whose Spearman correlation with a on cubic hosts is at
            least 0.2 in magnitude.

Figure 1 is not drawn here. It plots individual literature measurements that are
not in this repository; only the fitted curve is, in Data/benchmarks/UNUC.csv.

Figure 2 uses the same plotting function and resolution as the published figure.
Figures 3 and 4 follow the published layout, with two differences: the legend of
Figure 3 uses the model names from the text, and the feature labels of Figure 4
are horizontal, so that each sits level with its own bar.

USAGE
-----
    python scripts/make_figures.py [--out-dir Results/figures/manuscript]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import config  # noqa: E402
from engine.reporting import plot_actual_vs_predicted  # noqa: E402

RESULTS = _REPO_ROOT / "Results"
DEFAULT_OUT = RESULTS / "figures" / "manuscript"

# Figure 3 series in legend order, with the colours of the published figure.
SERIES = [
    ("lin",  "Linear Regression", "gray"),
    ("rf1",  "Lumped RF",         "orange"),
    ("rf2",  "Independent RF",    "black"),
    ("gbr1", "Lumped GBR",        "blue"),
    ("gbr2", "Independent GBR",   "purple"),
]


def _bold_ticks(ax, size):
    ax.tick_params(labelsize=size)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def figure2(out_dir: Path) -> Path:
    cv = pd.read_csv(RESULTS / "metrics" / "cv_predictions" / "cv_predictions_rf1.csv")
    params = ["a", "b", "c"]
    y_true = cv[[f"{p}_true" for p in params]].set_axis(params, axis=1)
    y_pred = cv[[f"{p}_pred" for p in params]].set_axis([f"{p}_pred" for p in params], axis=1)

    cubic = cv["crystal_system"] == "cubic"
    err = y_pred[cubic].to_numpy() - y_true[cubic].to_numpy()
    mse = (err ** 2).mean(axis=0)
    mae = np.abs(err).mean(axis=0)
    ss_tot = ((y_true[cubic] - y_true[cubic].mean()) ** 2).sum().to_numpy()
    r2 = 1 - (err ** 2).sum(axis=0) / ss_tot

    plot_actual_vs_predicted(
        y_true, y_pred, config.MODEL_FILES["rf1"][1],
        mse_vals=mse, mae_vals=mae, r2_vals=r2,
        df=cv[["crystal_system"]], col="crystal_system", labels=["cubic"],
        outdir=out_dir, stat_prefix="Cubic ",
        dataset_label="Cubic Only Dataset Size", dataset_size=int(cubic.sum()),
    )
    out = out_dir / "figure2_cross_validation_rf1.png"
    (out_dir / "lumped_rf.png").replace(out)
    return out


def figure3(out_dir: Path) -> Path:
    unuc = pd.read_csv(RESULTS / "benchmarks" / "UNUC" / "predictions.csv")
    ceo2 = pd.read_csv(RESULTS / "benchmarks" / "CeO2Nd2O3" / "predictions.csv")
    systems = [
        (unuc, 1 - unuc["y"], "C Content $\\mathbf{x}$",
         "True vs Predicted\nLattice Parameter of UN$_{\\mathbf{1-x}}$C$_{\\mathbf{x}}$"),
        (ceo2, ceo2["composition"].str.extract(r"Nd([\d.]+)")[0].astype(float).fillna(0.0),
         "Nd Content $\\mathbf{x}$",
         "True vs Predicted\nLattice Parameter of Ce$_{\\mathbf{1-x}}$Nd$_{\\mathbf{x}}$O$_{\\mathbf{2}}$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2))
    for ax, (df, x, xlabel, title) in zip(axes, systems):
        ax.scatter(x, df["a_true"], marker="o", s=60, color="red", label="True $\\mathbf{a}$",
                   zorder=3)
        for key, name, colour in SERIES:
            ax.scatter(x, df[f"a_pred_{key}"], marker="s", s=50, color=colour,
                       label=f"{name} $\\mathbf{{a}}$", zorder=2)
        ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
        ax.set_ylabel("Lattice Parameter a (Å)", fontsize=18, fontweight="bold")
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.grid(alpha=0.3)
        _bold_ticks(ax, 15)

    handles, labels = axes[0].get_legend_handles_labels()
    # Column-major order puts True a over Linear Regression, as published.
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               prop={"size": 16, "weight": "bold"})
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    out = out_dir / "figure3_validation_systems.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def figure4(out_dir: Path) -> Path:
    s = pd.read_csv(RESULTS / "metrics" / "feature_spearman_cubic.csv")
    s = s[s["in_figure"]].sort_values("abs_rho", ascending=False, kind="stable")
    n = len(s)
    colours = plt.get_cmap("coolwarm")(np.linspace(0, 1, n))
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.barh(y, s["spearman_rho"], color=colours, height=0.8)
    ax.set_yticks(y, s["feature"])
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Spearman Correlation with Lattice Parameter", fontsize=16, fontweight="bold")
    ax.set_ylabel("Input Feature Label", fontsize=16, fontweight="bold")
    ax.set_title("Spearman Correlation of Features for Lattice Parameter\n"
                 "of Cubic Crystal Systems (≥ 0.2 or ≤ −0.2)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=14)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    fig.tight_layout()
    out = out_dir / "figure4_spearman_cubic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help=f"Where to write the PNGs (default {DEFAULT_OUT.relative_to(_REPO_ROOT)})")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = [figure2(args.out_dir), figure3(args.out_dir), figure4(args.out_dir)]
    for p in written:
        print(f"[make_figures] wrote {p}")
    return written


if __name__ == "__main__":
    main()
