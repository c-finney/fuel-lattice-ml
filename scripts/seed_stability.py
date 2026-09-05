"""
seed_stability.py: how much of each model's benchmark result survives a change of
random seed.

WHY THIS EXISTS
---------------
The benchmark sets are small (23 and 7 compositions) and, more importantly, they
are extrapolation: fractional solid solutions like UN(0.6)C(0.4) are not in the
Materials Project training data. A single fit's Pearson r on 23 extrapolated
points is not by itself evidence about a model class, because refitting the same
estimator with a different seed can move it a long way.

That is not hypothetical here. Rebuilding gbr1 from the committed seed dataset
changed its U(N,C) correlation from -0.0972 to +0.8219, and changing only
random_state from 42 to 43 moved it again to +0.4209. Over seeds 42 to 46 it
spans +0.3764 to +0.8219, while rf1 over the same seeds stays within +0.8919 to
+0.9147 and gbr2 changes sign. Any statement of the form "model X inverts the
compositional trend" thus has to be checked against the spread across seeds
before it can be believed.

A standard deviation of exactly zero is a bug rather than a result: it means the
seed never reached the estimator that consumes it. See _set_seed.

This script measures that spread. It refits every model at several seeds, runs
both benchmarks against each fit, and reports mean and standard deviation of MAE,
slope and Pearson r. A model whose sign is stable across seeds is saying
something about the model; one whose sign flips is saying something about the fit.

Linear Regression has no random_state and is fitted once: its spread is zero by
construction, not by measurement.

The shipped binaries in Models/binaries/ are never touched. Each fit is written
to a temporary LATTICE_DATA_ROOT and deleted after it has been scored, so peak
extra disk is one model rather than seeds x models.

USAGE
-----
    python scripts/seed_stability.py [--seeds 42,43,44] [--models rf1,gbr1]

Expect hours rather than minutes: gbr1 alone is roughly 20 minutes per fit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402

from engine import config  # noqa: E402
from engine.train_models import model_estimators, prepare_training_frame  # noqa: E402

BENCHMARKS = {
    "UNUC":      ("U(N,C)",     "Data/benchmarks/UNUC.csv"),
    "CeO2Nd2O3": ("(Ce,Nd)O2",  "Data/benchmarks/CeO2Nd2O3Vals.csv"),
}

# Deterministic estimators, fitted once because refitting proves nothing.
NO_SEED = {"lin"}


def _set_seed(est, seed: int) -> bool:
    """
    Set random_state wherever it actually governs the fit. False if there is none.

    The outer estimator is tried FIRST, and the order matters. A
    RandomForestRegressor exposes `.estimator`, but that is the
    DecisionTreeRegressor template rather than a wrapped model, and the forest
    overwrites each tree's random_state from its own during fit. Descending into
    it therefore sets a seed that is silently discarded, and the forest refits at
    its default seed every time. That failure is quiet and looks like a result:
    a standard deviation of exactly zero across every seed.

    Only a meta-estimator with no random_state of its own, such as
    MultiOutputRegressor, should have the seed pushed into its inner estimator.
    """
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
        return True
    inner = getattr(est, "estimator", None)
    if inner is not None and "random_state" in inner.get_params():
        inner.set_params(random_state=seed)
        return True
    return False


def _set_jobs(est, n_jobs: int) -> None:
    inner = getattr(est, "estimator", None)
    if inner is not None and "n_jobs" in inner.get_params():
        inner.set_params(n_jobs=n_jobs)
        if "n_jobs" in est.get_params():
            est.set_params(n_jobs=1)
    elif "n_jobs" in est.get_params():
        est.set_params(n_jobs=n_jobs)


def _score(df: pd.DataFrame, key: str) -> dict | None:
    col = f"a_pred_{key}"
    if col not in df.columns or "a_true" not in df.columns:
        return None
    valid = df[["a_true", col]].dropna()
    if len(valid) < 2:
        return None
    t = valid["a_true"].to_numpy()
    p = valid[col].to_numpy()
    return {
        "n":         len(valid),
        "MAE":       float(np.abs(p - t).mean()),
        "slope":     float(np.polyfit(t, p, 1)[0]),
        "pearson_r": float(np.corrcoef(t, p)[0, 1]),
    }


def run(keys: list[str], seeds: list[int], n_jobs: int = -1) -> pd.DataFrame:
    _, X, Y, _ = prepare_training_frame()
    print(f"[seed_stability] training frame {X.shape[0]} rows x {X.shape[1]} features")

    records = []
    with tempfile.TemporaryDirectory(prefix="seed-stability-") as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "Models" / "binaries").mkdir(parents=True)
        (tmp_root / "Dataset").mkdir(parents=True)
        env = {**os.environ, "LATTICE_DATA_ROOT": str(tmp_root)}

        bin_dir = tmp_root / "Models" / "binaries"

        for key in keys:
            fname = config.MODEL_FILES[key][0]
            target = bin_dir / fname
            use_seeds = [seeds[0]] if key in NO_SEED else seeds

            # artifacts.check_prereqs() refuses to predict unless at least one
            # REPORTABLE model is present, so link the shipped binaries for every
            # OTHER key into this root. They are never loaded, because --models
            # restricts prediction to the key under test; they only satisfy the
            # prereq. The key under test is deliberately NOT linked: joblib.dump
            # would write straight through the symlink into Models/binaries/ and
            # overwrite the shipped artifact.
            for other, (other_name, _) in config.MODEL_FILES.items():
                link = bin_dir / other_name
                if link.exists() or link.is_symlink():
                    link.unlink()
                if other == key:
                    continue
                shipped = config.REPO_ROOT / "Models" / "binaries" / other_name
                if shipped.exists():
                    link.symlink_to(shipped)

            for seed in use_seeds:
                est = model_estimators()[key]
                seeded = _set_seed(est, seed)
                _set_jobs(est, n_jobs)
                print(f"[seed_stability] {key} seed={seed if seeded else 'n/a'} fitting…",
                      flush=True)
                est.fit(X, Y)
                _set_jobs(est, None)
                assert not target.is_symlink(), (
                    f"{target} is a symlink; dumping would overwrite the shipped binary"
                )
                joblib.dump(est, target)

                for bench, (system, csv) in BENCHMARKS.items():
                    out = tmp_root / f"out_{key}_{seed}_{bench}"
                    subprocess.run(
                        [sys.executable, "cli.py", "predict", "--csv", csv,
                         "--models", key, "--include-baseline", "--out-dir", str(out)],
                        cwd=_ROOT, env=env, check=True, capture_output=True,
                    )
                    scored = _score(pd.read_csv(out / "predictions.csv"), key)
                    if scored is None:
                        print(f"[seed_stability]   {bench}: no prediction column, skipped")
                        continue
                    records.append({
                        "model_key":  key,
                        "model_name": config.MODEL_FILES[key][1],
                        "benchmark":  bench,
                        "system":     system,
                        "seed":       seed if seeded else None,
                        "seeded":     seeded,
                        **scored,
                    })
                    print(f"[seed_stability]   {bench}: MAE {scored['MAE']:.4f} Å, "
                          f"r {scored['pearson_r']:+.4f}", flush=True)
                target.unlink(missing_ok=True)

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model_key", "model_name", "benchmark", "system", "seeded"])
    out = g.agg(
        n_fits=("seed", "size"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        slope_mean=("slope", "mean"), slope_sd=("slope", "std"),
        r_mean=("pearson_r", "mean"), r_sd=("pearson_r", "std"),
        r_min=("pearson_r", "min"), r_max=("pearson_r", "max"),
    ).reset_index()
    out["sign_stable"] = (np.sign(out["r_min"]) == np.sign(out["r_max"]))
    return out


def merge_existing(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fold a partial run into whatever is already on disk.

    Each (model, seed, benchmark) measurement is independent of every other, so
    re-running one model reproduces exactly the rows a full run would produce for
    it. Without this, re-running a single model would silently discard the other
    four, which is a bad way to lose two hours of fitting.
    """
    path = config.REPO_ROOT / "Results" / "benchmarks" / "seed_stability.csv"
    if not path.exists():
        return raw
    old = pd.read_csv(path)
    keep = old[~old["model_key"].isin(raw["model_key"].unique())]
    if len(keep):
        print(f"[seed_stability] merging {len(keep)} existing rows for "
              f"{sorted(keep['model_key'].unique())}")
    return pd.concat([keep, raw], ignore_index=True)


def write_report(raw: pd.DataFrame, summ: pd.DataFrame, seeds: list[int]) -> None:
    out_dir = config.REPO_ROOT / "Results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "seed_stability.csv", index=False)

    lines: list[str] = []
    w = lines.append
    w("# Seed stability of the benchmark metrics")
    w("")
    w("GENERATED by `scripts/seed_stability.py`. Do not hand-edit.")
    w("")
    w(f"Every model refitted at seeds {', '.join(str(s) for s in seeds)} and scored on both")
    w("solid-solution benchmarks. Linear Regression has no `random_state`, so it is fitted")
    w("once and its spread is zero by construction rather than by measurement.")
    w("")
    w("`sign_stable` is whether Pearson r kept the same sign across every fit. Where it did")
    w("not, no claim about the direction of the compositional trend can rest on that model,")
    w("because the direction is a property of the fit rather than of the model.")
    w("")
    for bench in sorted(summ["benchmark"].unique()):
        sub = summ[summ.benchmark == bench].sort_values("model_key")
        system = sub["system"].iloc[0]
        w(f"## {system} (`{bench}`)")
        w("")
        w("| model | fits | MAE (Å) | slope | Pearson r | r range | sign stable |")
        w("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            sd = lambda v: "" if pd.isna(v) else f" ± {v:.4f}"  # noqa: E731
            w(f"| {r.model_name} (`{r.model_key}`) | {int(r.n_fits)} "
              f"| {r.MAE_mean:.4f}{sd(r.MAE_sd)} "
              f"| {r.slope_mean:.3f}{sd(r.slope_sd)} "
              f"| {r.r_mean:+.4f}{sd(r.r_sd)} "
              f"| {r.r_min:+.4f} to {r.r_max:+.4f} "
              f"| {'yes' if r.sign_stable else '**NO**'} |")
        w("")
    (out_dir / "seed_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[seed_stability] wrote {out_dir/'seed_stability.csv'}")
    print(f"[seed_stability] wrote {out_dir/'seed_stability.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--models", default=",".join(config.SCOREABLE))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--append", action="store_true",
                    help="Keep existing rows for models not covered by this run, instead "
                         "of replacing the whole file. Use when re-running one model.")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    keys = [k.strip() for k in args.models.split(",")]
    raw = run(keys, seeds, args.n_jobs)
    if raw.empty:
        raise SystemExit("[seed_stability] nothing scored")
    if args.append:
        raw = merge_existing(raw)
    summ = summarize(raw)
    write_report(raw, summ, seeds)
    print()
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
