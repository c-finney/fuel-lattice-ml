---
name: lattice-train
description: >
  Train the lattice-parameter regression models on the featurized dataset (full-dataset
  fit). Fast mode trains Dependent RF (rf1) only — the model already shipped in this
  repository. Use when the user asks to train, retrain, or fit the lattice-parameter
  models, or invokes /lattice-train.
---

Train the lattice-parameter models via `cli.py` at the repository root. Models are
**fit on the full filtered dataset** (a,b,c ≤ 10 Å) and saved to `Models/binaries/`.
Cross-validation is **not** done here — it is a separate metrics-only step
(`/lattice-evaluate`).

## Usage

- `/lattice-train` or `/lattice-train fast` — train **Dependent RF only** (`rf1`, the
  model already shipped in this repository as `Models/binaries/DependentRFModel.joblib`;
  default).
- `/lattice-train full` — train all reportable models plus Linear Regression (LR is
  stored but never shown in prediction output).
- `/lattice-train models=rf1,gbr1` — train an explicit subset.

Model keys: `rf1`=Dependent RF (headline — currently the only trained model in this
repository), `rf2`=Independent RF, `gbr1`=Dependent GBR (XGBoost), `gbr2`=Independent
GBR (HistGBR), `lin`=Linear Regression (full only, never reported).

## Procedure

1. **Precondition:** the featurized dataset must exist. Run `python cli.py status --json`;
   if the dataset/feature-labels are missing, tell the user to run `/lattice-build`
   first (or offer to run it — with the committed seed present it's ~9 minutes, not the
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

- Final models are full-dataset fits (`.fit(X, Y)`) — these are what prediction loads.
- Use `/lattice-evaluate` to estimate model accuracy via 5-fold cross-validation; that
  step refits its own fold models and never touches these saved binaries.
- Overwriting `Models/binaries/DependentRFModel.joblib` here means it no longer matches
  `Models/DependentRFModel/metrics.json` (generated from a specific past cross-validation
  run) until you re-run `/lattice-evaluate` — mention this if the user retrains `rf1`.
