# Independent GBR — HistGBR (`gbr2`)

**Status: trained and 5-fold cross-validated.** Binary:
`Models/binaries/ScikitLearnGBRModel.joblib` (22,965,965 bytes — the smallest reportable model).

## Description

A `MultiOutputRegressor` wrapping one independent `HistGradientBoostingRegressor` per
lattice parameter (a, b, c) — the gradient-boosted analogue of `rf2`'s "independent"
design.

## Hyperparameters

See `params.json`: `max_iter=1800`, `learning_rate=0.05`, `max_depth=10`,
`max_features=0.8`, `random_state=42`.

Note: an earlier, unmaintained exploratory notebook
(`exploratory/FuelLatticeParameterModelCreation_TrainTest.ipynb`) trains this same
model with `max_iter=1500` instead of `1800` — a real discrepancy between that notebook
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

- **It does not track the U(N,C) compositional trend — it inverts it.** On
  `Data/benchmarks/UNUC.csv`, `gbr2` has a Pearson r of **−0.2696** and a slope of **−0.426**
  against the experimental values: the predicted lattice parameter moves the *wrong way* with
  composition. Its raw MAE (0.048023 Å) understates this badly — it is landing in the right
  numeric neighbourhood while getting the physics backwards. **Do not use `gbr2` for U(N,C)
  interpolation.** (`rf2`, by contrast, has r = 0.9405 there.) See
  `Results/benchmarks/basis_check.md`.
- **The training labels are DFT, the benchmarks are experimental — different quantities.**
  Every training lattice parameter is Materials-Project DFT-relaxed geometry (MP's
  `theoretical: False` means the structure was *observed*, not that its lattice parameters were
  *measured*). A benchmark MAE against experimental `a_true` therefore contains the
  DFT-vs-experiment discrepancy on top of model error. It is material-specific and changes
  sign: DFT − experiment is −0.006621 Å (UN), −0.022736 Å (UC), **+0.057365 Å** (CeO2). No
  correction is shipped — only 3 of 9 curated hosts have an in-repo experimental value and the
  sign flips across them. Compare models with the offset-invariant slope / Pearson r from
  `scripts/basis_check.py`.
- **Validated primarily on cubic hosts.** R²_cubic (0.977992) far exceeds R²_all (0.886472).
  `predict_one()` warns on non-cubic hosts.
- **Out-of-domain elements** absent from the training features degrade accuracy;
  `predict_one()` warns when detected.
- **Pickle format risk.** Raw `joblib`/pickle artifact — see `Models/README.md`.

## To retrain

```bash
python cli.py train --models gbr2
```
