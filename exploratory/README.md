# exploratory/

**Unmaintained.** The test suite does not run these five notebooks; it only scans them
for local paths and the revoked API key. They do not import `engine/` and are not
guaranteed to run on the pinned dependency versions in `requirements.txt`. They are kept
for the historical record of what was tried.

- **`FuelLatticeParameterModelCreation_TrainTest.ipynb`**, a train/test-split sibling
  of the core `FuelLatticeParameterModelCreation_CrossVal.ipynb`. Trains `gbr2`
  (`HistGradientBoostingRegressor`) with `max_iter=1500`; `engine/train_models.py`,
  which the CrossVal notebook calls, sets 1800, so this file is not a source for the
  shipped hyperparameters. Its stored metrics come from an earlier dataset snapshot
  (64,785 rows) and a single 80/20 split; cite `Results/metrics/ModelMetrics_CrossVal.csv`
  instead.
- **`FuelLatticeParameterNNModelCreation.ipynb`**, a PyTorch/Optuna neural-network
  exploration. Persists no artifacts (no model file, no metrics file).
- **`FuelLatticeParameterSymbolicExpression.ipynb`**: despite the name, fits only
  `sklearn.linear_model.LinearRegression`; no symbolic regression library is used.
- **`FuelLatticeParameterUTest.ipynb`**, a uranium hold-out check: a 600-tree random
  forest trained on every compound without U and scored on the U-containing compounds.
  It uses crystal-system one-hot encoding on the full feature list, so its feature set
  differs from `rf1`'s. Persists no artifacts.
- **`FuelLatticeParameterVisualization.ipynb`**, a uranium AB/AB2 lattice-parameter
  survey. It queries the Materials Project through `pymatgen.ext.matproj.MPRester`,
  where the shipped `engine/mp_client.py` uses `mp_api.client.MPRester`. Its API key and
  file paths are redacted placeholders, and it needs `adjustText`, which
  `requirements.txt` does not list.

To extend the modeling work, start from `engine/` and the four core notebooks at the
repository root.
