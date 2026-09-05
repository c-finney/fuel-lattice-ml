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
| a | 0.113556 | 0.983007 | 0.300814 | 0.903586 |
| b | 0.113306 | 0.983016 | 0.309928 | 0.893973 |
| c | 0.123666 | 0.980368 | 0.384835 | 0.862923 |

**`gbr1` is the most accurate model in the repository on cross-validation**, winning 11 of
the 12 metric×parameter cells against `rf1`/`rf2`/`gbr2`. Its only loss is `MAE_all` for
parameter c (`rf1` 0.381954 vs 0.384835 Å), on the all-systems set. On `MAE_cubic`, the
regime relevant to fluorite and rocksalt fuels, it leads `rf1` 0.113556 vs 0.121701 Å **while
being 77× smaller** (51 MB vs 3.97 GB).

> [!warning] **Do not read that as "gbr1 is the best model."** Cross-validation accuracy
> and repeatability on out-of-domain compositions are different properties. On the U(N,C)
> benchmark `gbr1`'s Pearson r spans **+0.3764 to +0.8219** across seeds 42 to 46, a
> standard deviation of 0.2092 against `rf1`'s 0.0084 on the same benchmark, so a single
> figure for it carries a wide margin. `rf1` leads `config.HEADLINE_PREF` on that
> repeatability; `gbr1` sits 3rd. See `Results/benchmarks/seed_stability.md`.

## Limitations

- **Cross-validation accuracy does not carry over to the solid-solution benchmarks.** At
  the shipped seed of 42 this model scores Pearson r = +0.8219 on
  `Data/benchmarks/UNUC.csv`, against a cross-validated `MAE_cubic` that leads every other
  model here. Refitting at seeds 43 to 46 gives +0.4209, +0.8092, +0.6209 and +0.3764, a
  mean of +0.6099 with a standard deviation of 0.2092, where `rf1` on the same benchmark
  sits at +0.9028 ± 0.0084 and is 25 times tighter. The benchmark is extrapolation, since
  these fractional solid solutions are absent from the training data, and an 1800-round
  depth-10 ensemble varies more there than a bagged forest does. Quote a `gbr1` benchmark
  correlation with the seed it came from, and do not choose `gbr1` for U(N,C) interpolation
  on the strength of its CV score. See `Results/benchmarks/seed_stability.md`.
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
- **Validated primarily on cubic hosts.** R²_cubic (0.983007) is far above R²_all (0.903586);
  non-cubic predictions carry materially more uncertainty. `predict_one()` warns on non-cubic hosts.
- **Out-of-domain elements** absent from the training features degrade accuracy;
  `predict_one()` warns when detected.
- **Pickle format risk.** Raw `joblib`/pickle artifact, as `Models/README.md` explains.

## To retrain

```bash
python cli.py train --models gbr1
```
