---
name: lattice-train
description: >
  Train the lattice-parameter regression models on the featurized dataset (full-dataset
  fit). Fast mode trains Lumped RF (rf1) only, which is the headline model of the five in this
  repository. Use when the user asks to train, retrain, or fit the lattice-parameter
  models, or invokes /lattice-train.
---

Train the lattice-parameter models via `cli.py` at the repository root. Models are
**fit on the full filtered dataset** (a,b,c ≤ 10 Å) and saved to `Models/binaries/`.
Cross-validation is **not** done here, being a separate metrics-only step
(`/lattice-evaluate`).

## Usage

- `/lattice-train` or `/lattice-train fast`: train **Lumped RF only** (`rf1`, the
  model already shipped in this repository as `Models/binaries/LumpedRFModel.joblib`;
  default).
- `/lattice-train full`: train all reportable models plus Linear Regression (LR is
  stored but never shown in prediction output).
- `/lattice-train models=rf1,gbr1`: train an explicit subset.

Always pass `--n-jobs -1` to `cli.py train`. It is **not** the default: with `n_jobs`
unset, `RandomForestRegressor` falls back to a single core, and rf1 takes many times
longer than the ETA suggests. Parallelism does not change the fitted forest, because
scikit-learn draws each tree's seed from `random_state` before dispatch.

Model keys: `rf1`=Lumped RF (the headline model, chosen on combined cross-validation
and benchmark evidence rather than on any single metric), `rf2`=Independent RF,
`gbr1`=Lumped GBR (XGBoost), `gbr2`=Independent GBR (HistGBR), `lin`=Linear
Regression (full only, never reported). All five are trained and archived.

## Procedure

1. **Precondition:** the featurized dataset must exist. Run `python cli.py status --json`;
   if the dataset/feature-labels are missing, tell the user to run `/lattice-build`
   first, or offer to run it, since with the committed seed present it's ~9 minutes and not the
   old multi-hour estimate), and stop.

2. **State the ETA and confirm:** fast (`rf1`) ≈ minutes; full (including 1800-estimator
   XGBoost + HistGBR) ≈ tens of minutes or more. Ask **yes/no** for `full`.

3. **Run** (background for `full`; foreground is fine for `fast`):
   ```
   python cli.py train --fast
   #   or  python cli.py train --full
   #   or  python cli.py train --models rf1,gbr1
   #   optional: --n-jobs N to cap CPU/memory
   ```

4. **On completion**, report which models were written to `Models/binaries/` and that
   `/lattice-predict` can now use them. Remind the user prediction works with any
   subset of non-LR models available.

## Notes

- Final models are full-dataset fits (`.fit(X, Y)`), and these are what prediction loads.
- Use `/lattice-evaluate` to estimate model accuracy via 5-fold cross-validation; that
  step refits its own fold models and never touches these saved binaries.
- Overwriting `Models/binaries/LumpedRFModel.joblib` here means it no longer matches
  `Models/LumpedRFModel/metrics.json` (generated from a specific past cross-validation
  run) until you re-run `/lattice-evaluate`, so mention this if the user retrains `rf1`.
