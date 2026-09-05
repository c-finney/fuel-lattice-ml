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
   RF (`rf1`) is the headline model on stability rather than accuracy, and **not** because
   it wins cross-validation, which it does not: `gbr1` has the better `MAE_cubic` at
   0.113556 Å against 0.121701 Å. What separates them is the spread of the benchmark
   metrics across random seeds, where `rf1` sits at Pearson r +0.9028 ± 0.0084 on U(N,C)
   against `gbr1`'s +0.6099 ± 0.2092 and `gbr2`'s +0.0768 ± 0.3945. Do not call any model
   "most accurate" without saying on which axis, and quote a benchmark correlation with
   the seed it came from. See `Results/benchmarks/seed_stability.md`. Point to the plots written under
   `Results/figures/`, and to the per-entry predictions under
   `Results/metrics/cv_predictions/`.

## Notes

- Metrics-only: this command does **not** save prediction models.
- `evaluate_cv.py` deliberately injects parallelism into each estimator rather than
  into `cross_val_predict` itself (`n_jobs=1` for the CV split, `n_jobs=-1` on the
  estimator), because the naive approach would hold up to 5 fully-grown 600-tree forests in
  memory simultaneously (~20 GB peak for rf1). Don't "optimize" this back.
