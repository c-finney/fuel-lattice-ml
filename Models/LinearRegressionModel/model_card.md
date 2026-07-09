# Linear Regression (`lin`)

**Status: not trained.** No binary exists for this model in this repository.

## Description

A `MultiOutputRegressor` wrapping plain `sklearn.linear_model.LinearRegression`,
one independent linear fit per lattice parameter (a, b, c). This is a **baseline
sanity check only** — it is trained solely under `cli.py train --full` and is **never**
shown in prediction output (see `config.REPORTABLE`, which deliberately excludes it).

## To train this model

```bash
python cli.py train --full   # trains all 5 models including lin
# or
python cli.py train --models lin
```
