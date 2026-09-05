# Linear Regression (`lin`)

**Status: trained and 5-fold cross-validated.** Binary:
`Models/binaries/LinearRegressionModel.joblib`, 17,153 bytes, by a wide margin the
smallest artifact here.

## Description

A `MultiOutputRegressor` wrapping plain `sklearn.linear_model.LinearRegression`, one
independent linear fit per lattice parameter (a, b, c). This is a **baseline sanity
check**, trained only under `cli.py train --full` and suppressed from prediction output
by default, because a model this far off should never be mistaken for a usable
prediction. `config.REPORTABLE` excludes it and `config.SCOREABLE` adds it back for the
benchmark scoring paths.

Passing `--include-baseline` to `cli.py predict` opts it in. That flag exists because
the benchmark table in the accompanying manuscript reports this model, so its numbers
have to be reproducible from this repository.

## Hyperparameters

See `params.json`, which records that there are none, since this is `LinearRegression()`
at scikit-learn defaults wrapped for multi-output.

## Training data

`engine/train_models.py`'s `prepare_training_frame()`: Materials Project structures
filtered to a, b, c ≤ 10 Å, 145 features, 64,128 rows, which is identical to the frame
every other model here was fitted on.

## Metrics

5-fold `cross_val_predict`, from `Results/metrics/ModelMetrics_CrossVal.csv`:

| param | MAE_cubic (Å) | R²_cubic | MAE_all (Å) | R²_all |
|---|---|---|---|---|
| a | 1.046069 | 0.431484 | 1.058157 | 0.416516 |
| b | 1.054062 | 0.418105 | 1.064247 | 0.418121 |
| c | 1.038633 | 0.435423 | 1.147785 | 0.348827 |

An order of magnitude worse than any ensemble model, which is the point of keeping it.
Lattice parameter depends on composition in a way a linear fit cannot represent, and
R²_cubic of 0.43 against 0.98 for the ensembles is the measurement of that.

## The one place it is interesting

On the solid-solution benchmarks, absolute error and trend fidelity separate sharply,
and this model is the clearest illustration in the repository:

| benchmark | MAE vs exp (Å) | slope | Pearson r |
|---|---|---|---|
| U(N,C) | 0.472301 | -1.048 | **-0.9629** |
| (Ce,Nd)O₂ | 0.907920 | **1.030** | 0.8418 |

On U(N,C) it has the largest absolute error of the five models and the largest
correlation magnitude of any of them, reproducing the size of the compositional
dependence while inverting its sign. On (Ce,Nd)O₂ its slope of 1.030 is the closest to
unity anything here achieves, and almost all of its 0.9079 Å error is a constant offset.
A model can be badly wrong and still track the trend, or land close and track nothing,
and those are different failures.

## Limitations

- **It is not a predictor.** Nothing in this card should be read as a case for using it.
- **The training labels are DFT, the benchmarks are experimental, and they are different
  quantities.** See `Results/benchmarks/basis_check.md`, which applies to every model here.
- **Pickle format risk.** Raw `joblib`/pickle artifact, as `Models/README.md` explains,
  though at 17 KB this is the one model small enough to inspect by hand.

## To retrain

```bash
python cli.py train --models lin --n-jobs -1
```
