# Independent GBR, HistGBR (`gbr2`)

**Status: trained and 5-fold cross-validated.** Binary:
`Models/binaries/ScikitLearnGBRModel.joblib`, 22,965,965 bytes, the smallest reportable model.

## Description

A `MultiOutputRegressor` wrapping one independent `HistGradientBoostingRegressor` per
lattice parameter (a, b, c), the gradient-boosted analogue of `rf2`'s "independent"
design.

## Hyperparameters

See `params.json`: `max_iter=1800`, `learning_rate=0.05`, `max_depth=10`,
`max_features=0.8`, `random_state=42`.

Note: an earlier, unmaintained exploratory notebook
(`exploratory/FuelLatticeParameterModelCreation_TrainTest.ipynb`) trains this same
model with `max_iter=1500` instead of `1800`, a real discrepancy between that notebook
and this repository's canonical hyperparameters, never reconciled. `params.json` and
`engine/train_models.py` are authoritative; the exploratory notebook is not.

## Training data

`engine/train_models.py`'s `prepare_training_frame()`: Materials Project structures
filtered to a,b,c ≤ 10 Å, 145 features, 64,128 rows.

## Metrics

5-fold `cross_val_predict`, from `Results/metrics/ModelMetrics_CrossVal.csv`:

| param | MAE_cubic (Å) | R²_cubic | MAE_all (Å) | R²_all |
|---|---|---|---|---|
| a | 0.151228 | 0.977992 | 0.364360 | 0.886472 |
| b | 0.152303 | 0.977945 | 0.380820 | 0.873556 |
| c | 0.174036 | 0.972340 | 0.485335 | 0.823700 |

`gbr2` has the **worst MAE_cubic of the four reportable models** (0.151228 Å, vs `gbr1`
0.113556). It is compact (23 MB) but not accurate.

## Limitations

- **It does not follow the U(N,C) compositional trend.** At the shipped seed of 42 it
  gives Pearson r = **−0.2696** and a slope of **−0.426** against the experimental values,
  so the predicted lattice parameter moves the wrong way with composition, and its raw MAE
  of 0.048023 Å understates that by landing in the right numeric neighbourhood. `rf1` on
  the same benchmark gives +0.9012. **Do not use `gbr2` for U(N,C) interpolation.**

  The value above describes the deposited binary at `random_state=42`. Quote it from that
  binary rather than from a local refit: boosting fits 1,800 successive rounds against the
  previous round's residuals, so floating-point differences between machines compound from
  round to round instead of cancelling as they do across a forest's independent trees.
- **The training labels are DFT and the benchmarks are experimental. They are
  different quantities.**
  Every training lattice parameter is Materials-Project DFT-relaxed geometry (MP's
  `theoretical: False` means the structure was *observed*, not that its lattice parameters were
  *measured*). A benchmark MAE against experimental `a_true` thus contains the
  DFT-vs-experiment discrepancy on top of model error. It is material-specific and changes
  sign: DFT − experiment is −0.006621 Å (UN), −0.022736 Å (UC), **+0.057365 Å** (CeO2). No
  correction is shipped, because only 3 of 9 curated hosts have an in-repo
  experimental value and the
  sign flips across them. Compare models with the offset-invariant slope / Pearson r from
  `scripts/basis_check.py`.
- **Validated primarily on cubic hosts.** R²_cubic (0.977992) far exceeds R²_all (0.886472).
  `predict_one()` warns on non-cubic hosts.
- **Out-of-domain elements** absent from the training features degrade accuracy;
  `predict_one()` warns when detected.
- **Pickle format risk.** Raw `joblib`/pickle artifact, as `Models/README.md` explains.

## To retrain

```bash
python cli.py train --models gbr2
```
