# Models/

## Layout

```
Models/
  MANIFEST.json                 committed: provenance, versions, hashes, Zenodo URIs
  feature_labels/                committed: the model input contract
    FeatureLabels.joblib          503 labels (build-time; NO 'nelements')
    FeatureLabels.json            human-readable mirror of the above
    ML_FeatureLabels.joblib       145 labels (train-time; final model input)
    ML_FeatureLabels.json         human-readable mirror of the above
  <ModelName>/
    model_card.md                committed: intended use, training data, metrics, limitations
    params.json                  committed: exact hyperparameters (parity-test fixture)
    metrics.json                 committed, ONLY for trained models
  binaries/                      GITIGNORED: fetched via scripts/fetch_models.py
    LumpedRFModel.joblib          3.97 GB, the headline model
    IndependentRFModel.joblib     9.74 GB, the largest artifact here
    XGBoostGBRModel.joblib          51 MB
    ScikitLearnGBRModel.joblib      23 MB
    LinearRegressionModel.joblib    17 KB, baseline only, never reported
```

## All five models are trained

Each of the five directories carries a `model_card.md`, a `params.json` and a
`metrics.json`, and each has a binary in the Zenodo deposit. The comparison behind
Tables 1 and 2 of the accompanying manuscript is `Results/metrics/ModelMetrics_CrossVal.csv`
for cross-validation and `Results/benchmarks/basis_check.csv` for the two solid-solution
systems.

`rf1` is the headline model in `config.HEADLINE_PREF`. `gbr1` has the better
cross-validated `MAE_cubic`, 0.113556 Å against 0.121701 Å, and is 77 times smaller.

The forests are preferred because they reproduce the direction of the compositional
dependence on U(N,C). Pearson r is +0.9012 (`rf1`) and +0.9405 (`rf2`); both boosted models
are negative, −0.0972 (`gbr1`) and −0.2696 (`gbr2`), which puts the lattice parameter
falling as carbon substitutes for nitrogen. Between the two forests, `rf2` is marginally
more accurate but is last on CV R²_cubic and weighs 9.74 GB. See
`Results/benchmarks/basis_check.md`.

These figures describe the deposited binaries at `random_state=42`, which is what
`scripts/fetch_models.py` downloads and what the manuscript reports. Take the boosted
models from the deposit rather than from a local refit: boosting fits 1,800 successive
rounds against the previous round's residuals, so floating-point differences between
machines compound from round to round, and `gbr1` can refit to a measurably different
model. A forest averages 600 independently built trees and cancels them instead.

## Fetching the binary

The model binaries are not stored in git, as the root `.gitignore` enforces.
Fetch them with:

```bash
python scripts/fetch_models.py
```

This needs no token and no account: the binaries live in the open Zenodo deposit recorded
in `MANIFEST.json`, and the script talks to it over plain HTTPS. It verifies the downloaded
file's SHA-256 against `Models/MANIFEST.json` before the file is unpickled. `joblib.load`
executes arbitrary code embedded in a pickle, so that hash check is the only barrier
between a compromised download and code execution. Serializing to `skops` or ONNX would
remove the risk and was not done.

## Loading the model: `mmap_mode="r"` is required

`engine/predict.py`'s `_load_model()` always loads with `joblib.load(path, mmap_mode="r")`.
This is not a performance tweak: `joblib`'s default (non-mmap) reader reconstructs
every ndarray by reading it flat and assigning `.shape` in place, which NumPy >= 2.5
deprecated. For a 600-tree forest that fires ~1,800 `DeprecationWarning`s on every load.
The memmap path never executes that assignment, so it is silent, ~3.3x faster
(measured: 10.0 s -> 3.0 s), and produces bitwise-identical predictions. It does **not**
reduce memory, because scikit-learn's `Tree.__setstate__` copies the memmapped arrays into
C-owned buffers regardless, so predicting with `rf1` still needs about 4.5 GB of free RAM.

The model binary must stay uncompressed. `mmap_mode` is silently ignored by joblib for
compressed files, which would reinstate both the warnings and the slower load. There is no
`compress_model.py` in this repository, since that script was considered and rejected for
this reason.

Version pins matter for pickle compatibility, not for the warnings above. `requirements.txt`
pins `scikit-learn==1.9.*` and `joblib==1.5.*`, the versions the deposited models were
trained and pickled under, as recorded in `MANIFEST.json`. `numpy` is not upper-bounded:
pinning it below 2.5 was tested and breaks scipy's `sparse.linalg.eigen.arpack` import.
`scripts/fetch_models.py` compares the running `scikit-learn` version against the manifest
and warns on a mismatch before unpickling.

## Regenerating the JSON label mirrors

`engine/build_dataset.py` and `engine/train_models.py` write both the `.joblib` and `.json`
forms of the feature label lists together, into these same tracked paths. A clean rebuild on
the same Materials Project snapshot reproduces byte-identical pickles, since a pickled list
of `str` is deterministic, so `git status` stays clean. A diff here after a rebuild means the
feature contract moved, and should be reviewed rather than committed.
