# Independent RF (`rf2`)

**Status: not trained.** No binary exists for this model in this repository, and no
cross-validation metrics have been generated for it.

## Description

A `MultiOutputRegressor` wrapping one independent `RandomForestRegressor` per lattice
parameter (a, b, c), as opposed to `rf1` (Dependent RF), which uses a single native
multi-output forest where every tree predicts all three parameters jointly.

## Hyperparameters

See `params.json` — identical `RandomForestRegressor(n_estimators=600, random_state=42)`
per output, wrapped independently.

## Training data

Same pipeline as `rf1` would use: `engine/train_models.py`'s `prepare_training_frame()`,
Materials Project structures filtered to a,b,c ≤ 10 Å, 145 features.

## To train this model

```bash
python cli.py train --models rf2
```
