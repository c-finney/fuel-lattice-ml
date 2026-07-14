"""
evaluate_cv.py — Cross-validation metrics (separate from training).

Runs KFold(5, shuffle=True, random_state=42) -> cross_val_predict per model.
Writes ModelMetrics_CrossVal.csv to METRICS_DIR.
Plots optionally to --out-dir.

CV metrics describe the hyperparameters and the training frame, not one
particular fitted forest — evaluate() refits fresh estimators per fold via
model_estimators() and never touches the saved .joblib binaries in
Models/binaries/.

CLI: python -m engine.evaluate_cv [--models rf1,rf2,...] [--out-dir DIR] [--n-jobs N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from engine import config
from engine.train_models import model_estimators, prepare_training_frame


def evaluate(
    keys: list[str] = config.REPORTABLE,
    out_dir: str | Path | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run 5-fold CV for *keys* and write ModelMetrics_CrossVal.csv.

    Parameters
    ----------
    keys    : model keys to evaluate (default: REPORTABLE = no LR)
    out_dir : if set, write pred-vs-true plots to this directory
    n_jobs  : parallelism, injected INTO each estimator (e.g. RandomForestRegressor's
              own n_jobs), never into cross_val_predict's own n_jobs.

              This is deliberate, not a style choice: cross_val_predict(n_jobs=-1)
              parallelizes across FOLDS, so 5 fully-grown 600-tree forests would be
              resident in memory simultaneously (~4 GB each for rf1 => ~20 GB peak).
              Parallelizing inside the estimator instead keeps exactly one forest
              in memory at a time, with comparable wall time (tree building
              parallelizes well within a single fit).

    Returns
    -------
    DataFrame with one row per model × parameter combo
    """
    _, X, Y, _ = prepare_training_frame()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    estimators = model_estimators()

    cubic_mask = X["cs_cubic"] == True  # noqa: E712

    records = []
    for key in keys:
        if key not in estimators:
            print(f"[evaluate] Unknown key '{key}', skipping")
            continue
        print(f"[evaluate] Cross-validating {config.MODEL_FILES[key][1]} ({key})…")
        est = estimators[key]

        # Push parallelism into the INNERMOST estimator, never into a meta-estimator
        # wrapper. (See also the n_jobs docstring above, about not parallelizing folds.)
        #
        # This ordering is load-bearing, and getting it backwards OOM-killed the rf2 CV
        # on a 31 GB box (2026-07-13):
        #
        #   MultiOutputRegressor(n_jobs=-1) fans the 3 outputs (a, b, c) out to loky
        #   PROCESSES. Each fits a complete 600-tree forest (~3.2 GB for rf2) and pickles
        #   it back, so the parent holds all three (~9.7 GB) while the workers still hold
        #   their own copies. Worse, the inner RandomForestRegressor is left at n_jobs=1,
        #   so we pay maximum memory for minimum parallelism.
        #
        #   RandomForestRegressor(n_jobs=-1) instead parallelizes TREE BUILDING with
        #   THREADS over a single shared copy of X. Same wall-clock win, a fraction of
        #   the RAM, and only one forest is resident at a time.
        #
        # The old code tested the wrapper FIRST (`if hasattr(est, "n_jobs")`), which
        # always matches for MultiOutputRegressor — so the inner branch never ran.
        inner = getattr(est, "estimator", None)
        if inner is not None and hasattr(inner, "n_jobs"):
            inner.set_params(n_jobs=n_jobs)      # rf2, lin: parallelize inside (threads)
            if hasattr(est, "n_jobs"):
                est.set_params(n_jobs=1)         # keep the wrapper sequential
        elif hasattr(est, "n_jobs"):
            est.set_params(n_jobs=n_jobs)        # rf1, gbr1, gbr2: no inner n_jobs

        Y_pred = cross_val_predict(est, X, Y, cv=cv, n_jobs=1)

        for i, param in enumerate(["a", "b", "c"]):
            y_true = Y[param].values
            y_pred = Y_pred[:, i]

            mse  = mean_squared_error(y_true, y_pred,               multioutput="raw_values")[0]
            mae  = mean_absolute_error(y_true, y_pred,              multioutput="raw_values")[0]
            r2   = r2_score(y_true, y_pred,                         multioutput="raw_values")[0]

            # Cubic subset
            mse_c = mean_squared_error(y_true[cubic_mask], y_pred[cubic_mask], multioutput="raw_values")[0]
            mae_c = mean_absolute_error(y_true[cubic_mask], y_pred[cubic_mask], multioutput="raw_values")[0]
            r2_c  = r2_score(y_true[cubic_mask], y_pred[cubic_mask],           multioutput="raw_values")[0]

            records.append({
                "model_key":  key,
                "model_name": config.MODEL_FILES[key][1],
                "param":      param,
                "MSE_all":    mse,
                "MAE_all":    mae,
                "R2_all":     r2,
                "MSE_cubic":  mse_c,
                "MAE_cubic":  mae_c,
                "R2_cubic":   r2_c,
            })

        # Optional plot
        if out_dir:
            _plot_pred_vs_true(Y, Y_pred, key, Path(out_dir))

    metrics_df = pd.DataFrame(records)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / "ModelMetrics_CrossVal.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"[evaluate] Metrics saved to {out_path}")
    return metrics_df


def _plot_pred_vs_true(Y: pd.DataFrame, Y_pred: np.ndarray, key: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, param in enumerate(["a", "b", "c"]):
        ax = axes[i]
        ax.scatter(Y[param].values, Y_pred[:, i], alpha=0.3, s=5)
        lims = [min(Y[param].min(), Y_pred[:, i].min()),
                max(Y[param].max(), Y_pred[:, i].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"True {param} (Å)")
        ax.set_ylabel(f"Predicted {param} (Å)")
        ax.set_title(f"{config.MODEL_FILES[key][1]} — {param}")
    plt.tight_layout()
    out_path = out_dir / f"pred_vs_true_{key}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate]   Plot saved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate models via 5-fold cross-validation (metrics only)."
    )
    parser.add_argument("--models", type=str, default=",".join(config.REPORTABLE),
                        help="Comma-separated model keys (default: all REPORTABLE)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for pred-vs-true plots (optional)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallelism injected into each estimator (default -1 = all cores). "
                             "Never passed to cross_val_predict itself — see evaluate() docstring.")
    args = parser.parse_args(argv)

    keys = [k.strip() for k in args.models.split(",")]
    evaluate(keys=keys, out_dir=args.out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
