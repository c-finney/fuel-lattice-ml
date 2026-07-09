# Dependent GBR — XGBoost (`gbr1`)

**Status: not trained.** No binary exists for this model in this repository, and no
cross-validation metrics have been generated for it.

## Description

An `XGBRegressor` using `multi_strategy="multi_output_tree"`, i.e. a single native
multi-output gradient-boosted forest where each tree predicts all three lattice
parameters (a, b, c) jointly — analogous to `rf1`'s "dependent" design, but gradient
boosted rather than bagged.

## Hyperparameters

See `params.json`: 1,800 estimators, `learning_rate=0.05`, `max_depth=10`,
`subsample=0.8`, `tree_method="hist"`, `random_state=42`.

## Training data

Same pipeline as `rf1`: `engine/train_models.py`'s `prepare_training_frame()`,
Materials Project structures filtered to a,b,c ≤ 10 Å, 145 features.

## To train this model

```bash
python cli.py train --models gbr1
```
