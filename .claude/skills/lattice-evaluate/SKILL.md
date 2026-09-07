---
name: lattice-evaluate
description: >
  Estimate lattice-parameter model accuracy via 5-fold cross-validation. Metrics +
  plots only, and does not produce the prediction models themselves. Use when the user
  asks to evaluate, cross-validate, or check the accuracy of the lattice-parameter
  models, or invokes /lattice-evaluate.
---

Cross-validate the lattice-parameter models and write a metrics report, via `cli.py`
at the repository root. This estimates performance only; the models used for actual
prediction are the full-dataset fits produced by `/lattice-train`.

## Usage

- `/lattice-evaluate`: 5-fold CV for all reportable models.
- `/lattice-evaluate models=rf1,rf2`: CV for a subset.

## Procedure

1. **Precondition:** the featurized dataset must exist (run `/lattice-build` first if
   not). Check via `python cli.py status --json`.

2. **State the ETA and confirm:** CV refits each model 5x, so this is slower than a
   single train, especially for the gradient-boosting models. For `rf1` alone, budget
   roughly 1-1.5 hours on a typical machine. Ask **yes/no**; run in the **background**.

3. **Run:**
   ```
   python cli.py evaluate --models rf1,rf2,gbr1,gbr2 --out-dir Results/figures/
   ```
   (Omit `--models` to default to all reportable models, i.e. `config.REPORTABLE`.)

4. **Report:** summarize `Results/metrics/ModelMetrics_CrossVal.csv`, giving per-model
   MAE/MSE/R² for a, b, c, both overall and cubic-only. Quote the **cubic-only**
   numbers as the headline accuracy figures, because the models were validated
   primarily on cubic hosts and the all-systems numbers are materially worse. Lumped
   RF (`rf1`) is the headline model on fidelity to the compositional trend rather than
   accuracy, and **not** because it wins cross-validation, which it does not: `gbr1` has
   the better `MAE_cubic` at 0.113556 Å against 0.121701 Å. What separates them is that on
   the U(N,C) benchmark `rf1` gives Pearson r +0.9012 while both boosted models invert the
   trend, at −0.0972 (`gbr1`) and −0.2696 (`gbr2`). Do not call any model "most accurate"
   without saying on which axis. See `Results/benchmarks/basis_check.md`, and
   `Results/benchmarks/seed_stability.md` for how far a change of seed moves each model.
   Point to the plots written under `Results/figures/`, and to the per-entry predictions
   under `Results/metrics/cv_predictions/`, whose provenance caveat is in
   `Results/README.md`.

## Notes

- Metrics-only: this command does **not** save prediction models.
- `evaluate_cv.py` deliberately injects parallelism into each estimator rather than
  into `cross_val_predict` itself (`n_jobs=1` for the CV split, `n_jobs=-1` on the
  estimator), because the naive approach would hold up to 5 fully-grown 600-tree forests in
  memory simultaneously (~20 GB peak for rf1). Don't "optimize" this back.
