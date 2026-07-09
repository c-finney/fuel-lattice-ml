# exploratory/

**Unmaintained.** These five notebooks are not covered by tests or CI, do not import
`engine/`, and are not guaranteed to run on the pinned dependency versions in
`requirements.txt`. They are kept for the historical record of what was tried.

- **`FuelLatticeParameterModelCreation_TrainTest.ipynb`** — a train/test-split sibling
  of the core `FuelLatticeParameterModelCreation_CrossVal.ipynb`. Trains `gbr2`
  (`HistGradientBoostingRegressor`) with `max_iter=1500`, where the CrossVal notebook
  and `engine/train_models.py` both use `max_iter=1800`. This is a real discrepancy
  between the two notebooks, never reconciled — another reason this file is not the
  source of truth for the shipped model's hyperparameters.
- **`FuelLatticeParameterNNModelCreation.ipynb`** — a PyTorch/Optuna neural-network
  exploration. Persists no artifacts (no model file, no metrics file).
- **`FuelLatticeParameterSymbolicExpression.ipynb`** — despite the name, fits only
  `sklearn.linear_model.LinearRegression`; no symbolic regression library is used.
- **`FuelLatticeParameterUTest.ipynb`** — trains an `rf1`-equivalent forest for ad hoc
  checks. Persists no artifacts.
- **`FuelLatticeParameterVisualization.ipynb`** — uses the **deprecated**
  `pymatgen.ext.matproj.MPRester` (the shipped `engine/mp_client.py` uses the current
  `mp_api.client.MPRester`). This notebook will not run against the pymatgen version
  pinned in `requirements.txt`.

If you want to extend the modeling work, start from `engine/` and the four core
notebooks at the repository root, not from anything here.
