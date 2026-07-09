# Independent GBR — HistGBR (`gbr2`)

**Status: not trained.** No binary exists for this model in this repository, and no
cross-validation metrics have been generated for it.

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

Same pipeline as `rf1`: `engine/train_models.py`'s `prepare_training_frame()`,
Materials Project structures filtered to a,b,c ≤ 10 Å, 145 features.

## To train this model

```bash
python cli.py train --models gbr2
```
