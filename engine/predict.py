"""
predict.py — Notebook 3: prediction entry points.

- predict_one(composition, reference, models)  -> dict
- predict_csv(path, models, out_dir)           -> dict
- build_prediction_frame(rows)                 -> DataFrame

Key faithfulness notes:
  - Featurizes the FULL composition string (not reduced). The build path
    featurizes composition_reduced instead — this asymmetry is intentional:
    prediction needs to reflect the exact mixing fractions the user asked
    about (e.g. "Ce0.8343Nd0.1657O2"), not the reduced formula.
  - Loads ML_FeatureLabels.joblib; NEVER recomputes it at predict time — it
    is the model's fixed input contract, established once at train time.
  - Cubic -> single a = mean(a,b,c); non-cubic -> a,b,c separate.
  - Linear Regression is excluded from prediction output entirely (it is a
    baseline sanity check only, never a reported model).
  - Warns on out-of-domain elements.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from engine import artifacts, config, featurize, reporting
from engine.reference_resolver import parse_composition, resolve_reference


# ---------------------------------------------------------------------------
# Cached artifact loading
#
# Both loaders are cached because predict_csv calls predict_one once PER ROW,
# and predict_one previously reloaded these artifacts from disk on every
# call. For the 23-row Data/benchmarks/UNUC.csv benchmark that meant re-reading
# the 3.97 GB model file 23 times. The cache turns that into exactly one load
# for the lifetime of the process (e.g. the long-lived MCP server).
#
# maxsize MUST be >= the number of models a single run can touch. It was 2,
# which silently defeated the cache for any batch run over the default model
# set: predict_one iterates rf1, rf2, gbr1, gbr2 per row, so a 2-slot LRU
# evicts the model it will need again two lookups later and re-reads ~13.7 GB
# of forests EVERY row. Sized to the full model table so it cannot thrash.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=len(config.MODEL_FILES))
def _load_model(key: str):
    """
    Load a trained model by key, cached for the process lifetime.

    mmap_mode="r" is REQUIRED here, not an optimisation: joblib's default
    loader reconstructs every ndarray by reading it flat and then assigning
    `.shape` in place, which NumPy >= 2.5 deprecated — this fires ~3x per
    tree (1,800 warnings for a 600-tree forest). The memmap path never
    executes that assignment, so it is silent, and measured ~3.3x faster to
    load besides. Predictions are bitwise identical to a normal load.

    This ONLY works on an uncompressed .joblib file — joblib silently
    ignores mmap_mode for compressed files, which is why the shipped model
    binary is never compressed (see Models/README.md).
    """
    return joblib.load(config.model_path(key), mmap_mode="r")


@functools.lru_cache(maxsize=1)
def _load_ml_feature_labels() -> list[str]:
    """
    Load ML_FeatureLabels.joblib, cached for the process lifetime.

    No mmap_mode here — this artifact is a plain Python list of column-name
    strings, not an ndarray, so there is nothing to memory-map.
    """
    return joblib.load(config.ML_FEATURELABELS)


# ---------------------------------------------------------------------------
# Build prediction feature frame
# ---------------------------------------------------------------------------

def build_prediction_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Build a one-row-per-composition feature DataFrame ready for model.predict().

    Each row dict must contain:
      - composition  : full composition string
      - crystal_system, spacegroup_num, is_centrosymmetric, n_symmetry_ops, nsites
        (from resolved reference)

    Steps:
      1. Build base DataFrame from rows
      2. OHE crystal_system + spacegroup_num with fixed config sets
      3. Create comp_obj from FULL composition string (not reduced)
      4. Featurize with same MultipleFeaturizer as build
      5. Select ML_FeatureLabels
    """
    df = pd.DataFrame(rows)

    # OHE
    df_cs = featurize.ohe_crystal_system(df)
    df_sg = featurize.ohe_spacegroup(df)
    df = pd.concat([df.reset_index(drop=True),
                    df_cs.reset_index(drop=True),
                    df_sg.reset_index(drop=True)], axis=1)

    # Composition objects from FULL composition string (not reduced — see
    # module docstring)
    from pymatgen.core import Composition
    df["comp_obj"] = [Composition(row["composition"]) for row in rows]

    # Featurize
    fzer = featurize.make_featurizer()
    df = featurize.featurize_compositions(df, "comp_obj", fzer)
    df = df.drop(columns=["comp_obj"], errors="ignore")

    # Load ML feature labels and select exactly those columns
    feature_labels: list[str] = _load_ml_feature_labels()

    # Hard check: every required label must be present. A missing column is
    # a hard KeyError, not a silent zero-fill — silently filling would
    # produce a confident-looking but meaningless prediction.
    missing_cols = [c for c in feature_labels if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Prediction frame is missing {len(missing_cols)} feature columns: "
            f"{missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}. "
            "This usually means featurization failed or the feature label set is mismatched."
        )

    return df[feature_labels]


# ---------------------------------------------------------------------------
# Domain guard
# ---------------------------------------------------------------------------

def _out_of_domain_warnings(composition: str, feature_labels: list[str]) -> list[str]:
    """
    Warn if any element in *composition* is not represented in the training features.
    Element fraction features have the element symbol as their label (e.g. 'U', 'N').
    """
    warnings = []
    try:
        from pymatgen.core import Composition
        comp = Composition(composition)
        el_symbols = {el.symbol for el in comp.elements}
        # Element fraction labels are just element symbols that appear in feature_labels
        train_elements = {lbl for lbl in feature_labels if len(lbl) <= 3 and lbl[0].isupper()}
        out_of_domain = el_symbols - train_elements
        if out_of_domain:
            warnings.append(
                f"Out-of-domain elements detected: {sorted(out_of_domain)}. "
                "These elements were absent/empty in the training set. "
                "Prediction accuracy may be degraded."
            )
    except Exception:
        pass
    return warnings


# ---------------------------------------------------------------------------
# predict_one
# ---------------------------------------------------------------------------

def predict_one(
    composition: str,
    reference: str | None = None,
    models: list[str] | None = None,
) -> dict:
    """
    Predict lattice parameters for a single composition.

    Parameters
    ----------
    composition : solid-solution formula ("UN0.5C0.5", "U1 N0.5 C0.5", …)
    reference   : optional mp-id or formula to force the reference structure
    models      : optional list of model keys; default = all available REPORTABLE models

    Returns
    -------
    dict with keys:
      status     : "ok" | "needs_build" | "needs_reference"
      headline   : {model, values: {a[,b,c]}, units}
      table      : [{model_key, model_name, a[,b,c]}]
      reference  : {mp_id, crystal_system, …, basis, source}
      warnings   : [str]
    """
    # Prereq check
    prereq = artifacts.check_prereqs("predict")
    if not prereq["ok"]:
        return {
            "status":   "needs_build",
            "missing":  prereq["missing"],
            "build_eta": prereq["build_eta"],
            "train_eta": prereq["train_eta"],
        }

    prereq_warnings = list(prereq.get("warnings", []))

    av = prereq["available_models"]
    if models:
        av = [k for k in models if k in av and k != "lin"]
    if not av:
        return {
            "status":  "needs_build",
            "missing": ["at least one trained model"],
            "build_eta": prereq["build_eta"],
            "train_eta": prereq["train_eta"],
        }

    # Reference resolution
    try:
        comp = parse_composition(composition)
    except Exception as exc:
        return {
            "status": "needs_reference",
            "candidates": [],
            "reason": f"Cannot parse composition '{composition}': {exc}",
        }

    ref_result = resolve_reference(comp, explicit_ref=reference)
    if ref_result.get("status") != "ok":
        return ref_result  # needs_reference

    crystal_system = ref_result.get("crystal_system", "")

    # Build prediction frame
    rows = [{
        "composition":        composition,
        "crystal_system":     crystal_system,
        "spacegroup_num":     ref_result.get("spacegroup_num"),
        "is_centrosymmetric": ref_result.get("is_centrosymmetric"),
        "n_symmetry_ops":     ref_result.get("n_symmetry_ops"),
        "nsites":             ref_result.get("nsites"),
    }]
    try:
        X_pred = build_prediction_frame(rows)
    except Exception as exc:
        return {
            "status": "needs_reference",
            "candidates": [],
            "reason": f"Featurization failed: {exc}",
        }

    # Load feature labels for domain check
    try:
        feature_labels = _load_ml_feature_labels()
    except Exception:
        feature_labels = []

    warnings = prereq_warnings + _out_of_domain_warnings(composition, feature_labels)
    if crystal_system.lower() != "cubic":
        warnings.append(
            "Non-cubic crystal system: a, b, c reported separately. "
            "Model accuracy was validated primarily on cubic systems."
        )

    # Predict with each available model
    table = []
    all_preds: dict[str, dict[str, float]] = {}

    for key in av:
        try:
            model = _load_model(key)
            raw = model.predict(X_pred)  # shape (1, 3)
            a, b, c = float(raw[0, 0]), float(raw[0, 1]), float(raw[0, 2])
            collapsed = reporting.collapse({"a": a, "b": b, "c": c}, crystal_system)
            all_preds[key] = collapsed

            row = {
                "model_key":  key,
                "model_name": config.MODEL_FILES[key][1],
            }
            row.update({k: f"{v:.4f}" for k, v in collapsed.items()})
            table.append(row)
        except Exception as exc:
            warnings.append(f"Model {key} failed: {exc}")

    if not all_preds:
        return {
            "status":   "needs_build",
            "missing":  ["all models failed at predict time"],
            "warnings": warnings,
            "build_eta": prereq["build_eta"],
            "train_eta": prereq["train_eta"],
        }

    # Headline model
    hl_key = reporting.headline_model(list(all_preds.keys()))
    hl_vals = all_preds[hl_key]

    # Reference provenance dict (strip internal keys)
    ref_out = {k: v for k, v in ref_result.items() if not k.startswith("_")}
    ref_out.pop("status", None)

    return {
        "status":    "ok",
        "headline":  {
            "model":  hl_key,
            "values": hl_vals,
            "units":  "Å",
        },
        "table":     table,
        "reference": ref_out,
        "warnings":  warnings,
    }


# ---------------------------------------------------------------------------
# predict_csv
# ---------------------------------------------------------------------------

def predict_csv(
    path: str,
    models: list[str] | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """
    Batch prediction from a CSV file.

    Expected CSV columns:
      composition  (required)
      ref_mp-id    (optional — overrides auto-resolution per row)
      y            (optional — mixing fraction, informational)
      a_true       (optional — adds MAE/MSE columns to output)

    Returns dict with keys: rows, predictions_path, warnings
    """
    df_in = pd.read_csv(path)
    results = []

    for _, row in df_in.iterrows():
        comp_str = str(row["composition"])
        ref = row.get("ref_mp-id") if "ref_mp-id" in row.index else None
        if pd.isna(ref):
            ref = None

        pred = predict_one(comp_str, reference=str(ref) if ref else None, models=models)

        result_row = {"composition": comp_str}
        if "y" in row.index:
            result_row["y"] = row["y"]
        if "a_true" in row.index:
            result_row["a_true"] = row["a_true"]

        if pred.get("status") == "ok":
            hl = pred["headline"]
            for param, val in hl["values"].items():
                # NOTE: this column name MUST vary by param (a/b/c), not just
                # by model. A prior version hardcoded "a_pred_{model}" here,
                # so for a non-cubic host (which reports a, b, AND c) every
                # iteration overwrote the same column and the last value
                # written (c) silently ended up under the "a_pred_*" name.
                # Cubic hosts never exposed this because reporting.collapse()
                # reduces them to a single "a" key before this loop runs.
                result_row[f"{param}_pred_{hl['model']}"] = val
            # All models
            for t in pred["table"]:
                k = t["model_key"]
                for param in ["a", "b", "c"]:
                    if param in t:
                        result_row[f"{param}_pred_{k}"] = float(t[param])
        else:
            result_row["status"] = pred.get("status")
            result_row["error"]  = pred.get("reason", "")

        results.append(result_row)

    out_df = pd.DataFrame(results)

    # Error metrics if a_true present.
    #
    # Scored for EVERY model that actually ran, not just rf1. predict_one already
    # predicts with all of them and writes an a_pred_<key> column each, but this
    # block used to hardcode a_pred_rf1 — so a batch run reported one model's
    # error and silently dropped the other three, which is precisely the number
    # you need to compare models on a benchmark.
    warnings = []
    metrics: dict[str, dict[str, float]] = {}
    if "a_true" in out_df.columns:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        for key in config.REPORTABLE:
            col = f"a_pred_{key}"
            if col not in out_df.columns:
                continue
            valid = out_df[["a_true", col]].dropna()
            if len(valid) == 0:
                continue
            mae = mean_absolute_error(valid["a_true"], valid[col])
            mse = mean_squared_error(valid["a_true"], valid[col])
            metrics[key] = {"mae": float(mae), "mse": float(mse), "n": int(len(valid))}
            warnings.append(
                f"{key} MAE={mae:.4f} Å, MSE={mse:.6f} Å² over {len(valid)} rows with a_true"
            )

    # Save output
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"
        out_df.to_csv(out_path, index=False)

        # Plot — all models, labelled (see reporting.plot_pred_vs_true)
        if "a_true" in out_df.columns:
            reporting.plot_pred_vs_true(
                out_df, out_dir,
                title=f"Predicted vs True Lattice Parameter $a$ — {Path(path).stem}",
            )
    else:
        out_path = None

    return {
        "rows": len(out_df),
        "predictions_path": str(out_path) if out_path else None,
        "metrics": metrics,
        "warnings": warnings,
        "dataframe": out_df,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict lattice parameters (Notebook 3)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--composition", type=str,
                      help="Single composition string (e.g. 'UN0.5C0.5')")
    mode.add_argument("--csv", type=str,
                      help="Path to CSV with 'composition' column")

    parser.add_argument("--reference", type=str, default=None,
                        help="Force reference structure (mp-id or formula)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (e.g. rf1,rf2)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for output CSV + plots")
    parser.add_argument("--json", action="store_true",
                        help="Print result as JSON (for MCP/script use)")
    args = parser.parse_args(argv)

    model_list = [k.strip() for k in args.models.split(",")] if args.models else None

    if args.composition:
        result = predict_one(args.composition, reference=args.reference, models=model_list)

        if args.out_dir and result.get("status") == "ok":
            # Write a copy to out_dir
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "prediction.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(reporting.format_markdown(result))
    else:
        result = predict_csv(args.csv, models=model_list, out_dir=args.out_dir)
        if args.json:
            # Remove non-serializable dataframe
            r = {k: v for k, v in result.items() if k != "dataframe"}
            print(json.dumps(r, indent=2))
        else:
            print(f"Batch prediction: {result['rows']} rows")
            for w in result["warnings"]:
                print(f"  ! {w}")
            if result.get("predictions_path"):
                print(f"  Output: {result['predictions_path']}")


if __name__ == "__main__":
    main()
