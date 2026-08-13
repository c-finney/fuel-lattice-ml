# Models/

## Layout

```
Models/
  MANIFEST.json                 committed — provenance, versions, hashes, HF URIs
  feature_labels/                committed — the model input contract
    FeatureLabels.joblib          503 labels (build-time; NO 'nelements')
    FeatureLabels.json            human-readable mirror of the above
    ML_FeatureLabels.joblib       145 labels (train-time; final model input)
    ML_FeatureLabels.json         human-readable mirror of the above
  <ModelName>/
    model_card.md                committed — intended use, training data, metrics, limitations
    params.json                  committed — exact hyperparameters (parity-test fixture)
    metrics.json                 committed, ONLY for trained models
  binaries/                      GITIGNORED — fetched via scripts/fetch_models.py
    LumpedRFModel.joblib        3.97 GB, the only PUBLISHED binary (all five are CV'd)
```

> **An empty `binaries/` is expected, not a broken checkout.** Binaries are gitignored and
> fetched on demand with `python scripts/fetch_models.py` (needs `HF_TOKEN` in `.env` while
> the HF repo is private). A fresh clone has `binaries/` holding only `.gitkeep` until you
> fetch, and every model-dependent script will fail until then.

## Only `rf1` (Lumped RF) is PUBLISHED — but all five are cross-validated

`Models/{IndependentRFModel,XGBoostGBRModel,ScikitLearnGBRModel,LinearRegressionModel}/`
carry a `model_card.md` and a `params.json`; only `LumpedRFModel/` also has a
`metrics.json` and a fetchable binary.

**A multi-model comparison HAS been run** — `Results/metrics/ModelMetrics_CrossVal.csv`
holds 5-fold CV rows for all five keys (× the `a`/`b`/`c` targets), and
`Results/benchmarks/basis_check.md` scores four of them on both solid-solution
benchmarks. The earlier text here said none had been run; that was stale.

> [!warning] **`MANIFEST.json` and `params.json` disagree on `"status"` (found 2026-08-12).**
> The manifest marks all five `"trained"`; four `params.json` files say `"not_trained"`.
> The two are using different senses of the word — *fit during CV* (all five) versus
> *persisted full-dataset artifact* (`rf1` only). Neither has been changed here: which
> field is authoritative is a release-review decision. Treat
> `Results/metrics/ModelMetrics_CrossVal.csv` as the measured ground truth.
> Note `tests/test_params_parity.py` pins `params.json` against
> `engine/train_models.model_estimators()`, so editing those status fields is not free.

## Fetching the binary

The model binary is **not** stored in git — see the root `.gitignore`. Fetch it with:

```bash
python scripts/fetch_models.py
```

This requires `HF_TOKEN` in `.env` (the HuggingFace repo is currently **private**; see
`PUBLICATION_CHECKLIST.md`). The script verifies the downloaded file's SHA-256 against
`Models/MANIFEST.json` **before** it is ever unpickled — `joblib.load` executes
arbitrary code embedded in the pickle, so hash verification is the only thing standing
between a compromised download and code execution. `skops`/ONNX serialization would
remove this risk entirely and is recorded here as future hardening, not yet done.

## Loading the model — `mmap_mode="r"` is required, not optional

`engine/predict.py`'s `_load_model()` always loads with `joblib.load(path, mmap_mode="r")`.
This is **not** a performance tweak: `joblib`'s default (non-mmap) reader reconstructs
every ndarray by reading it flat and assigning `.shape` in place, which NumPy >= 2.5
deprecated. For a 600-tree forest that fires ~1,800 `DeprecationWarning`s on every load.
The memmap path never executes that assignment, so it is silent, ~3.3x faster
(measured: 10.0 s -> 3.0 s), and produces bitwise-identical predictions. It does **not**
reduce memory — scikit-learn's `Tree.__setstate__` copies the memmapped arrays into
C-owned buffers regardless, so predicting with `rf1` still needs **~4.5 GB free RAM**.

**The model binary must stay UNCOMPRESSED.** `mmap_mode` is silently ignored by joblib
for compressed files, which would reinstate both the warnings and the slower load.
There is no `compress_model.py` in this repository — that script was considered and
rejected for exactly this reason.

**Version pins matter for pickle compatibility**, not for the warnings above:
`requirements.txt` pins `scikit-learn==1.9.*` and `joblib==1.5.*` (the exact versions
the shipped model was trained/pickled under — see `MANIFEST.json`). `numpy` is
deliberately **not** upper-bounded; pinning it below 2.5 was tested and breaks scipy's
`sparse.linalg.eigen.arpack` import entirely. `scripts/fetch_models.py` compares the
running `scikit-learn` version against the manifest and warns loudly on mismatch before
unpickling.

## Regenerating the JSON label mirrors

`engine/build_dataset.py` and `engine/train_models.py` write both the `.joblib` and
`.json` forms of the feature label lists together, into these same tracked paths. A
clean rebuild on the same Materials Project snapshot should reproduce byte-identical
pickles (a pickled list of `str` is deterministic) — `git status` should stay clean. If
a rebuild *does* produce a diff here, it is meaningful (the feature contract moved) and
must be reviewed before committing, never committed blindly.
