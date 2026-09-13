# Lumped GBR, XGBoost (`gbr1`)

**Status: trained and 5-fold cross-validated.** Binary: `Models/binaries/XGBoostGBRModel.joblib`
(51,341,788 bytes).

## Description

An `XGBRegressor` using `multi_strategy="multi_output_tree"`, i.e. a single native
multi-output gradient-boosted forest where each tree predicts all three lattice
parameters (a, b, c) jointly, analogous to `rf1`'s "lumped" design but gradient
boosted rather than bagged.

## Hyperparameters

See `params.json`: 1,800 estimators, `learning_rate=0.05`, `max_depth=10`,
`subsample=0.8`, `tree_method="hist"`, `random_state=42`.

## Training data

Same pipeline as `rf1`: `engine/train_models.py`'s `prepare_training_frame()`,
Materials Project structures filtered to a,b,c ≤ 10 Å, 145 features, 64,128 rows.

## Metrics

5-fold `cross_val_predict`, from `Results/metrics/ModelMetrics_CrossVal.csv`:

| param | MAE_cubic (Å) | R²_cubic | MAE_all (Å) | R²_all |
|---|---|---|---|---|
| a | 0.113433 | 0.982628 | 0.300566 | 0.903575 |
| b | 0.113509 | 0.982597 | 0.309441 | 0.893924 |
| c | 0.123930 | 0.979913 | 0.385168 | 0.862495 |

`gbr1` has the best cross-validation scores in the repository, taking 10 of the 12
metric×parameter cells against `rf1`, `rf2` and `gbr2`. Both losses are for parameter c,
where `rf1` gives `MAE_all` 0.381947 Å against 0.385168 Å and `MAE_cubic` 0.123894 Å against
0.123930 Å. On `MAE_cubic` for parameter a, the regime relevant to fluorite and rocksalt
fuels, it leads `rf1` 0.113433 against 0.121717 Å at 51 MB against 3.97 GB.

That does not make it the model to use. On the U(N,C) benchmark it returns Pearson r
= −0.0972 with a slope of −0.164, predicting the lattice parameter to fall as carbon
substitutes for nitrogen. `rf1` leads `config.HEADLINE_PREF` on that ground and `gbr1` sits
third. See `Results/benchmarks/basis_check.md`.

## Limitations

- **Cross-validation accuracy does not carry over to the solid-solution benchmarks.** This
  model has the best cross-validated `MAE_cubic` here and still inverts the U(N,C) trend, at
  Pearson r = −0.0972 and slope −0.164 against `rf1`'s +0.9012 and +1.240. The benchmark is
  extrapolation, since these fractional solid solutions are absent from the training data,
  and an 1800-round depth-10 ensemble does not extrapolate the way a bagged forest does.
  See `Results/benchmarks/basis_check.md`.
- **Take this model from the deposited binary rather than a local refit.** Boosting fits
  1,800 successive rounds, each against the previous round's residuals, so floating-point
  differences between machines compound from round to round instead of cancelling. Refitting
  `gbr1` on other hardware can produce a measurably different model, and this model's
  benchmark correlation is small enough to be moved by that. Everything reported here is the
  deposited binary at `random_state=42`. A forest averages 600 independently built trees and
  is not exposed to this.
- **The training labels are DFT and the benchmarks are experimental, which are different
  quantities.**
  Every training lattice parameter is Materials-Project DFT-relaxed geometry (MP's
  `theoretical: False` means the structure was *observed*, not that its lattice parameters were
  *measured*). A benchmark MAE against experimental `a_true` therefore contains the
  DFT-vs-experiment discrepancy on top of model error. It is material-specific and changes
  sign: DFT − experiment is −0.006621 Å (UN), −0.022736 Å (UC), +0.057365 Å (CeO2). No
  correction is shipped, because only 3 of 9 curated hosts have an in-repo experimental
  value and the sign flips across them. Compare models with the offset-invariant slope and
  Pearson r from `scripts/basis_check.py`.
- **Validated primarily on cubic hosts.** R²_cubic (0.982628) is well above R²_all
  (0.903575); non-cubic predictions carry more uncertainty. `predict_one()` warns on
  non-cubic hosts.
- **Out-of-domain elements** absent from the training features degrade accuracy;
  `predict_one()` warns when detected.
- **Pickle format risk.** Raw `joblib`/pickle artifact, as `Models/README.md` explains.

## To retrain

```bash
python cli.py train --models gbr1
```
