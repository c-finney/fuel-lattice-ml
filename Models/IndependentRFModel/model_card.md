# Independent RF (`rf2`)

Status: trained and 5-fold cross-validated. Binary:
`Models/binaries/IndependentRFModel.joblib`, 9,735,288,229 bytes, the largest artifact in
the repository at 2.5× `rf1`.

## Description

A `MultiOutputRegressor` wrapping one independent `RandomForestRegressor` per lattice
parameter (a, b, c), as opposed to `rf1` (Lumped RF), which uses a single native
multi-output forest where every tree predicts all three parameters jointly.

## Hyperparameters

See `params.json`. Each output gets an identical
`RandomForestRegressor(n_estimators=600, random_state=42)`, wrapped independently.

## Training data

`engine/train_models.py`'s `prepare_training_frame()`: Materials Project structures
filtered to a,b,c ≤ 10 Å, 145 features, 64,128 rows.

## Metrics

5-fold `cross_val_predict`, from `Results/metrics/ModelMetrics_CrossVal.csv`:

| param | MAE_cubic (Å) | R²_cubic | MAE_all (Å) | R²_all |
|---|---|---|---|---|
| a | 0.125516 | 0.976283 | 0.317180 | 0.886938 |
| b | 0.127465 | 0.974935 | 0.341422 | 0.870053 |
| c | 0.126418 | 0.974801 | 0.403132 | 0.843363 |

On cross-validation `rf2` is the lowest of the four reportable models by R²_cubic (0.976283,
against `gbr1`'s 0.983007) while being the largest. Independent per-output forests gain
nothing over `rf1`'s joint trees here, at 2.5× the size.

## Benchmark performance

Cross-validation does not rank these models the way the benchmarks do. On the metrics that
are invariant to the DFT/experiment label-basis offset (see Limitations), `rf2` has the
strongest correlation with the experimental compositional trend on both benchmarks:

| benchmark | n | Pearson r | slope | MAE vs exp (Å) |
|---|---|---|---|---|
| U(N,C), `UNUC.csv` | 23 | 0.9405 | 1.217 | 0.011661 |
| (Ce,Nd)O₂, `CeO2Nd2O3Vals.csv` | 7 | 0.9706 | 0.829 | 0.149348 |

Its high raw MAE on (Ce,Nd)O₂ is almost entirely a constant offset, a bias of +0.149348 Å.
With that removed it has the lowest scatter of any model there, 0.006921 Å against `gbr1`'s
0.036206. See `Results/benchmarks/basis_check.md`.

## Limitations

- **The training labels are DFT and the benchmarks are experimental, which are different
  quantities.**
  Every training lattice parameter is Materials-Project DFT-relaxed geometry (MP's
  `theoretical: False` means the structure was *observed*, not that its lattice parameters were
  *measured*). A benchmark MAE against experimental `a_true` therefore contains the
  DFT-vs-experiment discrepancy on top of model error. It is material-specific and changes
  sign: DFT − experiment is −0.006621 Å (UN), −0.022736 Å (UC), +0.057365 Å (CeO2), which
  accounts for most of `rf2`'s (Ce,Nd)O₂ bias. No correction is shipped: only 3 of the 9
  curated hosts have an in-repo experimental value, and the sign flips across those 3.
  Compare models with the offset-invariant slope and Pearson r from
  `scripts/basis_check.py`.
- **Size.** 9.74 GB, the largest artifact in the deposit. It is the lowest of the four on
  CV R²_cubic, so `rf1` is the better choice for most uses; `rf2` is deposited for
  completeness and for the benchmark correlations above.
- **Validated primarily on cubic hosts.** R²_cubic (0.976283) is well above R²_all
  (0.886938). `predict_one()` warns on non-cubic hosts.
- **Out-of-domain elements** absent from the training features degrade accuracy;
  `predict_one()` warns when detected.
- **Pickle format risk.** Raw `joblib`/pickle artifact, as `Models/README.md` explains.

## To retrain

```bash
python cli.py train --models rf2
```
