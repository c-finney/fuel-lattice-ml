"""
feature_correlations.py: Spearman correlation of every model input feature with
the cubic-system lattice parameter.

WHY THIS EXISTS
---------------
The feature-correlation figure in the accompanying manuscript was produced inside
FuelLatticeParameterModelPrediction.ipynb, which plotted the correlations and
never wrote them anywhere. That makes the figure impossible to check without
rerunning a notebook, so this script recomputes the same quantity and writes it
to disk as the point-level values behind that figure.

The computation matches the notebook exactly: rows restricted to
crystal_system == "cubic", Spearman correlation of each of the 145 ML feature
columns against the true lattice parameter `a`, sorted by absolute correlation.
The notebook then plots only the features with |rho| > 0.2; this script writes
ALL of them and flags that subset with the `in_figure` column, because the
threshold is a display choice and the discarded correlations are evidence too.

The notebook renders its bar chart with seaborn. Nothing here does, so the
values can be regenerated without that dependency.

USAGE
-----
    python scripts/feature_correlations.py

Requires Dataset/Training_Dataset.csv, which `cli.py build` and `cli.py train`
both write. Output goes to Results/metrics/feature_spearman_cubic.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd

from engine import config

THRESHOLD = 0.2


def compute() -> pd.DataFrame:
    if not config.DATASET_TRAINING.exists():
        raise SystemExit(
            f"{config.DATASET_TRAINING} not found. Run `python cli.py build --resume` "
            "first; the training frame is generated, not committed."
        )
    if not config.ML_FEATURELABELS.exists():
        raise SystemExit(f"{config.ML_FEATURELABELS} not found.")

    df = pd.read_csv(config.DATASET_TRAINING, low_memory=False)
    n_all = len(df)
    df = df[df["crystal_system"] == "cubic"]
    features: list[str] = joblib.load(config.ML_FEATURELABELS)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(
            f"{len(missing)} feature columns named in ML_FeatureLabels.joblib are absent "
            f"from the training frame, starting with {missing[:3]}. The feature contract "
            "and the dataset have diverged; do not trust this output."
        )

    corr = df[["a"] + features].corr(method="spearman")["a"].drop("a")

    out = corr.reset_index()
    out.columns = ["feature", "spearman_rho"]
    out["abs_rho"] = out["spearman_rho"].abs()
    out = out.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    out["in_figure"] = out["abs_rho"] > THRESHOLD
    out.insert(0, "n_cubic_rows", len(df))

    print(f"[correlations] {len(df)} cubic rows of {n_all} training rows")
    print(f"[correlations] {len(features)} features, "
          f"{int(out['in_figure'].sum())} with |rho| > {THRESHOLD}")
    return out


def main() -> None:
    out = compute()
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.METRICS_DIR / "feature_spearman_cubic.csv"
    out.to_csv(path, index=False)
    print(f"[correlations] Written to {path}")

    shown = out[out["in_figure"]]
    if len(shown):
        print(f"[correlations] Strongest positive: {shown.iloc[0]['feature']} "
              f"({shown.iloc[0]['spearman_rho']:+.4f})")
        neg = shown[shown["spearman_rho"] < 0]
        if len(neg):
            print(f"[correlations] Strongest negative: {neg.iloc[-1]['feature']} "
                  f"({neg.iloc[-1]['spearman_rho']:+.4f})")


if __name__ == "__main__":
    main()
