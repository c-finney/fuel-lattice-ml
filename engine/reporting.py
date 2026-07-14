"""
reporting.py — Headline model selection, cubic collapse, markdown formatting, plots.

Two distinct plotting functions live here — do not conflate them:
  - plot_pred_vs_true(df, out_dir)            : single-panel scatter for a
                                                  batch CSV prediction (used
                                                  by predict_csv / cli.py evaluate).
  - plot_actual_vs_predicted(...)              : 3-panel actual-vs-predicted
                                                  grid with an in-axes stats
                                                  box and optional colouring
                                                  by a categorical column
                                                  (e.g. crystal_system), used
                                                  by the core notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import config


# ---------------------------------------------------------------------------
# Headline model selection
# ---------------------------------------------------------------------------

def headline_model(available: list[str], metrics: dict | None = None) -> str:
    """
    Return the key of the headline model.

    Rule: rf1 if present; else first in HEADLINE_PREF order. rf1 is
    currently the ONLY trained model shipped in this repository — this is
    a fallback order, not a substantiated accuracy ranking (no comparative
    cross-validation across rf1/rf2/gbr1/gbr2 has been run; see
    Results/metrics/ModelMetrics_CrossVal.csv for rf1's own metrics).
    If *metrics* is supplied, could in future use accuracy ordering.
    """
    for key in config.HEADLINE_PREF:
        if key in available:
            return key
    raise ValueError(f"No headline model available from {available}")


# ---------------------------------------------------------------------------
# Cubic collapse
# ---------------------------------------------------------------------------

def collapse(pred_abc: dict[str, float], crystal_system: str) -> dict[str, float]:
    """
    For cubic: return {"a": mean(a, b, c)}.
    For non-cubic: return {"a": ..., "b": ..., "c": ...} as-is.

    *pred_abc* keys: "a", "b", "c" (single prediction, one model).
    """
    if crystal_system.lower() == "cubic":
        mean_a = float(np.mean([pred_abc["a"], pred_abc["b"], pred_abc["c"]]))
        return {"a": mean_a}
    return {k: float(v) for k, v in pred_abc.items()}


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def format_markdown(payload: dict) -> str:
    """
    Format a prediction payload as a markdown string.

    Expected payload keys (from predict_one):
      status, headline, table, reference, warnings
    """
    if payload.get("status") != "ok":
        reason = payload.get("reason", "unknown error")
        if payload["status"] == "needs_reference":
            return (
                f"**Reference required.** {reason}\n\n"
                f"Candidates: {', '.join(payload.get('candidates', []))}\n\n"
                "Re-run with `--reference <mp-id|formula>`."
            )
        if payload["status"] == "needs_build":
            missing = "\n- ".join(payload.get("missing", []))
            return (
                f"**Artifacts missing.** Cannot predict until built.\n\n"
                f"Missing:\n- {missing}\n\n"
                f"Build ETA: {payload.get('build_eta', 'unknown')}\n\n"
                "Run `/lattice-build` then `/lattice-train` first."
            )
        return f"**Error:** {reason}"

    lines = []

    # Headline
    hl = payload["headline"]
    model_name = config.MODEL_FILES.get(hl["model"], (hl["model"], hl["model"]))[1]
    if "a" in hl["values"] and "b" not in hl["values"]:
        lines.append(
            f"**Predicted lattice parameter (Å) — {model_name}:** "
            f"a = {hl['values']['a']:.4f} Å"
        )
    else:
        vals = ", ".join(f"{k} = {v:.4f}" for k, v in hl["values"].items())
        lines.append(f"**Predicted lattice parameters (Å) — {model_name}:** {vals} Å")
    lines.append("")

    # Table
    table = payload.get("table", [])
    if table:
        cols = list(table[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
            for row in table
        ]
        lines += [header, sep] + rows
        lines.append("")

    # Provenance
    ref = payload.get("reference", {})
    if ref:
        lines.append("**Reference structure provenance:**")
        mp_id  = ref.get("mp_id", "?")
        cs     = ref.get("crystal_system", "?")
        sg     = ref.get("spacegroup_num", "?")
        basis  = ref.get("basis", "?")
        source = ref.get("source", "?")
        lines.append(
            f"Host: `{mp_id}` | Crystal system: {cs} | Space group: {sg} | "
            f"Basis: {basis} | Source: {source}"
        )
        lines.append("")

    # Warnings
    warnings = payload.get("warnings", [])
    if warnings:
        lines.append("**Warnings:**")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pred-vs-true plot (single-panel, batch CSV mode)
# ---------------------------------------------------------------------------

# Per-model plot styling — one entry per REPORTABLE model, so every model that
# predicted is drawn and named. Keys must match config.MODEL_FILES.
_MODEL_STYLE: dict[str, tuple[str, str]] = {
    "rf1":  ("o", "tab:orange"),
    "rf2":  ("^", "tab:green"),
    "gbr1": ("D", "tab:blue"),
    "gbr2": ("v", "tab:purple"),
}


def plot_pred_vs_true(df: pd.DataFrame, out_dir: Path, title: str | None = None) -> Path:
    """
    Scatter plot of predicted vs true 'a' for a batch CSV prediction.

    Plots EVERY model present as its own labelled series, with a legend giving the
    model name and its MAE. It previously took the FIRST column matching
    ("pred" in c and "a" in c) and plotted that one series with no label — so the
    figure showed a single unidentified model and silently discarded the other
    three, which is useless for the thing this plot exists to support (comparing
    models on a benchmark).

    Expects: a_true, plus one or more a_pred_<model_key> columns.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "a_true" not in df.columns:
        return out_dir / "no_plot.txt"

    pred_cols = [
        (key, f"a_pred_{key}")
        for key in config.REPORTABLE
        if f"a_pred_{key}" in df.columns
    ]
    # Fall back to any a_pred_* column not covered by REPORTABLE (e.g. a headline alias)
    if not pred_cols:
        return out_dir / "no_plot.txt"

    fig, ax = plt.subplots(figsize=(7.5, 7))

    # Parity line spans every series, not just one
    all_vals = [df["a_true"]] + [df[c] for _, c in pred_cols]
    lo = min(float(s.min()) for s in all_vals)
    hi = max(float(s.max()) for s in all_vals)
    pad = 0.02 * (hi - lo) if hi > lo else 0.01
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "k--", linewidth=1.2, zorder=1, label="Perfect prediction ($y=x$)")

    for key, col in pred_cols:
        marker, color = _MODEL_STYLE.get(key, ("o", None))
        name = config.MODEL_FILES[key][1] if key in config.MODEL_FILES else key
        valid = df[["a_true", col]].dropna()
        label = f"{name} ({key})"
        if len(valid) > 0:
            mae = float((valid[col] - valid["a_true"]).abs().mean())
            label = f"{name} ({key}) — MAE {mae:.4f} Å"
        ax.scatter(
            df["a_true"], df[col],
            marker=marker, color=color, s=55, alpha=0.8,
            edgecolors="white", linewidths=0.5, zorder=2, label=label,
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True $a$ (Å)", fontsize=12)
    ax.set_ylabel("Predicted $a$ (Å)", fontsize=12)
    ax.set_title(title or "Predicted vs True Lattice Parameter $a$", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    plt.tight_layout()

    out_path = out_dir / "pred_vs_true.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Actual-vs-predicted plot (3-panel grid, notebook-facing)
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    Y_test: pd.DataFrame,
    Y_pred_df: pd.DataFrame,
    model_name: str,
    display_statistics: bool = True,
    mse_vals=None,
    mae_vals=None,
    r2_vals=None,
    df: pd.DataFrame | None = None,
    col: str | None = None,
    labels: list | None = None,
    display_others: bool = True,
    outdir: Path | str | None = None,
):
    """
    3-panel actual-vs-predicted scatter grid (one panel per lattice
    parameter a, b, c), with an optional in-axes MSE/MAE/R2 statistics box
    and optional colouring by a categorical column (e.g. crystal_system).

    Adapted from the core notebooks' PlotPredictions function.

    Parameters
    ----------
    Y_test, Y_pred_df : true and predicted values; Y_pred_df columns are
                         named "{param}_pred" for each param in Y_test.columns
    model_name         : used in the figure suptitle
    mse_vals/mae_vals/r2_vals : per-parameter arrays, indexed like Y_test.columns
    df, col, labels    : if given, colours points by df[col] restricted to
                         labels (+ "Other")
    outdir             : if given, saves to {outdir}/{slug}.png instead of
                         (or in addition to) an interactive show. This is the
                         mechanism that populates Results/figures/ — no
                         notebook calls savefig() directly.

    Returns
    -------
    Path to the saved figure, or None if outdir was not given.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    Y_test_df = Y_test.reset_index(drop=True)

    for i, param in enumerate(Y_test_df.columns):
        ax = axes[i]

        if col is not None and labels is not None:
            X_test_full = df.loc[Y_test.index].reset_index(drop=True)
            arr = np.asarray(X_test_full[col])
            missing = [f for f in labels if f not in arr]
            if missing:
                raise ValueError(f"The following labels are not found in column {col} of X_test_full: {missing}")

            labels_list = list(labels)
            cmap = plt.get_cmap("tab10")
            colors = {label: cmap(idx % cmap.N) for idx, label in enumerate(labels_list)}
            labels_list = ["Other"] + labels_list
            colors["Other"] = "gray"

            for label in labels_list:
                if label == "Other":
                    mask = ~np.isin(X_test_full[col], labels)
                    if not display_others or not mask.any():
                        continue
                else:
                    mask = X_test_full[col] == label

                ax.scatter(
                    Y_test_df.loc[mask, param],
                    Y_pred_df.loc[mask, f"{param}_pred"],
                    color=colors[label],
                    label=str(label).title(),
                    alpha=0.7,
                )

                line = [Y_test_df[param].min(), Y_test_df[param].max()]
                ax.plot(line, line, "k--", linewidth=1)

            leg = ax.legend(
                title=col.title() if col != "crystal_system" else "Crystal System",
                prop={"size": 16, "weight": "bold"},
            )
            leg.get_title().set_fontsize(18)
            leg.get_title().set_fontweight("bold")

        else:
            ax.scatter(Y_test_df[param], Y_pred_df[f"{param}_pred"], alpha=0.7)
            line = [Y_test_df[param].min(), Y_test_df[param].max()]
            ax.plot(line, line, "k--", linewidth=1)

        ax.set_xlabel("Actual Value", fontsize=20, fontweight="bold")
        ax.set_ylabel("Predicted Value", fontsize=20, fontweight="bold")
        ax.set_title(f"Lattice Parameter '{param}' (Å)", fontsize=22, fontweight="bold")
        ax.tick_params(labelsize=18)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

        if display_statistics and mse_vals is not None and mae_vals is not None and r2_vals is not None:
            stats = (
                f"MSE={mse_vals[i]:.4f}\n"
                f"MAE={mae_vals[i]:.4f}\n"
                f"R²={r2_vals[i]:.4f}"
            )
            if df is not None:
                stats += f"\nFull Dataset Size={len(df)}"
            ax.text(
                0.05, 0.95, stats,
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
                fontsize=16, fontweight="bold",
            )

    fig.suptitle(
        f"Predicted vs. Actual Lattice Parameter Using\n{model_name}",
        fontsize=24, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = outdir / f"{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.close(fig)
    return None
