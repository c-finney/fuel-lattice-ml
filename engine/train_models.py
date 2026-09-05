"""
train_models.py — Notebook 2: full-fit models.

Stages:
  1. prepare_training_frame()  — dedup, filter, feature-select, save Training_Dataset + ML_FeatureLabels
  2. model_estimators()        — returns exact hyperparameters (mirrored in
                                  Models/<name>/params.json and pinned by
                                  tests/test_params_parity.py)
  3. train(keys)               — .fit(X, Y) on full filtered data; joblib.dump; mark_stage

Final models are fit on the FULL filtered dataset (a,b,c <= 10 Å) — cross-validation
(evaluate_cv.py) is a separate metrics-only step that refits its own fold models and
never touches these saved binaries.

CLI:  python -m engine.train_models [--fast | --full | --models a,b] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import joblib
import pandas as pd

from engine import artifacts, config


# ---------------------------------------------------------------------------
# Preprocessing — shared between train and evaluate_cv
# ---------------------------------------------------------------------------

def prepare_training_frame():
    """
    Load Featurized CSV + FeatureLabels, apply dedup/filter/feature-selection,
    save Training_Dataset and ML_FeatureLabels.

    Returns (df, X, Y, filtered_labels)
    """
    # Load
    df = pd.read_csv(config.DATASET_FEATURIZED, low_memory=False)
    feature_labels: list[str] = joblib.load(config.FEATURELABELS)

    # Dedup single-element polymorphs (keep lowest formation_energy_per_atom)
    df = df.sort_values("formation_energy_per_atom", ascending=True)
    dup_mask = df.duplicated(
        subset=["composition_reduced", "spacegroup_num", "nsites"], keep="first"
    )
    keep_mask = ~((df["nelements"] == 1) & dup_mask)
    df = df.loc[keep_mask].reset_index(drop=True)
    df = df.sort_values("composition_reduced").reset_index(drop=True)

    # Lattice filter — training threshold (10 Å)
    df = df[(df[["a", "b", "c"]] <= config.TRAIN_THRESH).all(axis=1)].copy()

    # Drop sg_* OHE; keep scalar spacegroup_num (the scalar carries the same
    # information far more compactly, and the model was validated with it
    # present, so it stays even though the one-hot columns are removed)
    feature_labels = [c for c in feature_labels if c not in config.SG_LABELS]

    # Drop all-empty (all NaN or 0) feature columns
    empty_mask = ((df[feature_labels].isna()) | (df[feature_labels] == 0)).all()
    empty_cols = pd.Index(feature_labels)[empty_mask].tolist()
    feature_labels = [c for c in feature_labels if c not in empty_cols]
    df = df.drop(columns=empty_cols, errors="ignore")

    # Remove features with unwanted substrings (these min/max/range/mode
    # aggregate statistics and norms were found to hurt predictions for
    # dilute impurities), then add nsites back explicitly.
    to_remove = ["minimum", "maximum", "range", "mode", "norm", "max ionic"]
    filtered = [s for s in feature_labels if not any(sub in s for sub in to_remove)]
    filtered = filtered + ["nsites"]

    # Save Training_Dataset + ML_FeatureLabels (+ JSON mirror). These writers
    # target tracked files: a clean rebuild on the same MP snapshot produces
    # byte-identical pickles (a pickled list of str is deterministic), so
    # `git status` stays clean. If a diff DOES appear after a rebuild, it is
    # meaningful — the feature contract moved — and must be reviewed, never
    # blindly committed.
    config.DATASETS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_TRAINING, index=False)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(filtered, config.ML_FEATURELABELS)
    config.ML_FEATURELABELS.with_suffix(".json").write_text(
        json.dumps(filtered, indent=2), encoding="utf-8"
    )

    X = df[filtered]
    Y = df[["a", "b", "c"]]
    return df, X, Y, filtered


# ---------------------------------------------------------------------------
# Model registry — exact hyperparameters
#
# These are the contract. The authoritative copy for citation purposes is
# Models/<name>/params.json, and tests/test_params_parity.py fails CI if this
# function's output ever diverges from those files or from a hyperparameter
# reappearing hardcoded in a notebook.
# ---------------------------------------------------------------------------

def model_estimators() -> dict:
    """
    Return dict of key -> sklearn/xgb estimator with exact hyperparameters.

    NOTE: imports happen here (not at module top) so the module can be imported
    without having sklearn/xgboost installed (e.g. in tests that mock it).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    return {
        "rf1": RandomForestRegressor(n_estimators=600, random_state=42),
        "rf2": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=600, random_state=42)
        ),
        "gbr1": XGBRegressor(
            n_estimators=1800,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            tree_method="hist",
            multi_strategy="multi_output_tree",
            random_state=42,
        ),
        "gbr2": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=1800,
                learning_rate=0.05,
                max_depth=10,
                max_features=0.8,
                random_state=42,
            )
        ),
        "lin": MultiOutputRegressor(LinearRegression()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    keys: list[str],
    n_jobs: int | None = None,
    progress: Callable = print,
) -> dict:
    """
    Prepare the training frame, fit the requested models on the full dataset,
    dump joblib files, and mark_stage.

    Parameters
    ----------
    keys    : list of model keys from MODEL_FILES
    n_jobs  : if set, pass to RF/CV estimators where supported
    progress: status callback

    Returns
    -------
    dict with trained keys and artifact paths
    """
    progress("[train] Preparing training frame…")
    df, X, Y, filtered_labels = prepare_training_frame()
    progress(f"[train]   Training set: {len(df)} rows, {len(filtered_labels)} features")

    estimators = model_estimators()
    results = {}

    for key in keys:
        if key not in estimators:
            progress(f"[train]   Unknown model key '{key}', skipping")
            continue
        progress(f"[train] Fitting {config.MODEL_FILES[key][1]} ({key})…")

        est = estimators[key]

        # Push parallelism into the INNERMOST estimator, never into a meta-estimator
        # wrapper. engine/evaluate_cv.py documents the same ordering at length.
        #
        #   MultiOutputRegressor(n_jobs=-1) fans the three outputs out to loky
        #   PROCESSES, each fitting a complete 600-tree forest (~3.2 GB for rf2) and
        #   pickling it back, so the parent holds all three while the workers still
        #   hold their own copies. The inner RandomForestRegressor meanwhile stays at
        #   n_jobs=1, so this pays maximum memory for minimum parallelism.
        #
        #   RandomForestRegressor(n_jobs=-1) instead parallelizes TREE BUILDING with
        #   THREADS over one shared copy of X, for the same wall-clock win at a
        #   fraction of the memory.
        #
        # Testing the wrapper first always matches for MultiOutputRegressor, which
        # leaves the inner branch unreachable.
        inner = getattr(est, "estimator", None)
        if n_jobs is not None:
            if inner is not None and hasattr(inner, "n_jobs"):
                inner.set_params(n_jobs=n_jobs)       # rf2, lin
                if hasattr(est, "n_jobs"):
                    est.set_params(n_jobs=1)          # keep the wrapper sequential
            elif hasattr(est, "n_jobs"):
                est.set_params(n_jobs=n_jobs)         # rf1, gbr1, gbr2

        est.fit(X, Y)

        # Put n_jobs back before pickling. joblib writes the attribute into the
        # artifact, so without this the saved binary records whatever --n-jobs the
        # training machine happened to use, and two otherwise identical fits hash
        # differently. Parallelism is a property of the machine that did the
        # fitting, not of the model.
        #
        # This does NOT make a rebuild byte-identical to the binaries published in
        # 2026-07. A rebuild here came out 32 bytes larger than the recorded size,
        # and resetting n_jobs did not account for the difference; the cause was not
        # tracked down. What was checked instead is that the rebuilt models reproduce
        # the published metrics, which is in RELEASE.md.
        if n_jobs is not None:
            if inner is not None and hasattr(inner, "n_jobs"):
                inner.set_params(n_jobs=None)
            if hasattr(est, "n_jobs"):
                est.set_params(n_jobs=None)

        out_path = config.model_path(key)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, out_path)
        progress(f"[train]   Saved {out_path.name}")

        artifacts.mark_stage(f"train:{key}", features=len(filtered_labels),
                             rows=len(df), packages=artifacts.package_versions())
        results[key] = str(out_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train lattice-parameter models (Notebook 2, full-fit)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true",
                       help="Train Lumped RF only (rf1) — fastest")
    group.add_argument("--full", action="store_true",
                       help="Train all models (rf1, rf2, gbr1, gbr2, lin)")
    group.add_argument("--models", type=str,
                       help="Comma-separated model keys (e.g. rf1,gbr1)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel jobs for RF models")
    args = parser.parse_args(argv)

    if args.full:
        keys = list(config.MODEL_FILES.keys())
    elif args.models:
        keys = [k.strip() for k in args.models.split(",")]
    else:
        keys = ["rf1"]  # default = fast

    result = train(keys, n_jobs=args.n_jobs)
    print(f"[train] Done. Trained models: {list(result.keys())}")
    for k, p in result.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
