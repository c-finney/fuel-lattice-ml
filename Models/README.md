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

`rf1` is the headline model in `config.HEADLINE_PREF`, and the reason is worth stating
plainly because it is not the obvious one. `gbr1` beats it on cross-validation, at
`MAE_cubic` 0.113556 Å against 0.121701 Å while being 77 times smaller. What separates them
is repeatability on the benchmarks: refitted across seeds 42 to 46, `rf1` holds a U(N,C)
Pearson r of +0.9028 ± 0.0084 while `gbr1` ranges over +0.3764 to +0.8219 and `gbr2`'s
correlation changes sign depending on the seed. `rf2` is marginally the more accurate
forest but is last on CV R²_cubic while weighing 9.74 GB. `rf1` wins no single metric and
is the smaller of the two models that give the same answer twice. See
`Results/benchmarks/seed_stability.md`.

## Fetching the binary

The model binaries are **not** stored in git, as the root `.gitignore` enforces.
Fetch them with:

```bash
python scripts/fetch_models.py
```

This needs no token and no account: the binaries live in the open Zenodo deposit
recorded in `MANIFEST.json`, and the script talks to it over plain HTTPS. The script
verifies the downloaded file's SHA-256 against
`Models/MANIFEST.json` **before** it is ever unpickled, because `joblib.load` executes
arbitrary code embedded in the pickle, so hash verification is the only thing standing
between a compromised download and code execution. `skops`/ONNX serialization would
remove this risk entirely and is recorded here as future hardening, not yet done.

## Loading the model: `mmap_mode="r"` is required, not optional

`engine/predict.py`'s `_load_model()` always loads with `joblib.load(path, mmap_mode="r")`.
This is **not** a performance tweak: `joblib`'s default (non-mmap) reader reconstructs
every ndarray by reading it flat and assigning `.shape` in place, which NumPy >= 2.5
deprecated. For a 600-tree forest that fires ~1,800 `DeprecationWarning`s on every load.
The memmap path never executes that assignment, so it is silent, ~3.3x faster
(measured: 10.0 s -> 3.0 s), and produces bitwise-identical predictions. It does **not**
reduce memory, because scikit-learn's `Tree.__setstate__` copies the memmapped arrays into
C-owned buffers regardless, so predicting with `rf1` still needs **~4.5 GB free RAM**.

**The model binary must stay UNCOMPRESSED.** `mmap_mode` is silently ignored by joblib
for compressed files, which would reinstate both the warnings and the slower load.
There is no `compress_model.py` in this repository, since that script was considered and
rejected for exactly this reason.

**Version pins matter for pickle compatibility**, not for the warnings above:
`requirements.txt` pins `scikit-learn==1.9.*` and `joblib==1.5.*` (the exact versions
the shipped models were trained and pickled under, as recorded in `MANIFEST.json`). `numpy` is
deliberately **not** upper-bounded; pinning it below 2.5 was tested and breaks scipy's
`sparse.linalg.eigen.arpack` import entirely. `scripts/fetch_models.py` compares the
running `scikit-learn` version against the manifest and warns loudly on mismatch before
unpickling.

## Regenerating the JSON label mirrors

`engine/build_dataset.py` and `engine/train_models.py` write both the `.joblib` and
`.json` forms of the feature label lists together, into these same tracked paths. A
clean rebuild on the same Materials Project snapshot should reproduce byte-identical
pickles, since a pickled list of `str` is deterministic, so `git status` should stay clean. If
a rebuild *does* produce a diff here, it is meaningful (the feature contract moved) and
must be reviewed before committing, never committed blindly.
