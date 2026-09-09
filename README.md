# fuel-lattice-ml

Machine learning prediction of lattice parameters for nuclear fuel solid-solution
compositions, covering U(N,C), (Ce,Nd)O2 and other two-component mixed systems, trained
on Materials Project structures.

> ### This project is retired
>
> Development ended with the SULI appointment that produced it, on 1 August 2025. The
> repository is frozen at the state behind the accompanying manuscript and exists so
> that those results can be checked and reused. There is no maintenance, no support, and
> no response to issues or pull requests. Dependency versions will drift out from under
> this code and nobody will fix it. Fork it if you need it to keep working.
>
> ORNL software release number ORNL-CODE-XXXXXX. See RELEASE.md.

MIT licensed, archived at [doi:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).
The Materials Project data redistributed here, and the models derived from it, stay under
CC BY 4.0; see NOTICE.

---

## Quickstart

A fresh clone needs no Materials Project API key, because the post-query resume point ships
committed as `Data/MP_Dataset_Original_Trimmed.csv`, 21 MB:

```bash
git clone https://github.com/c-finney/fuel-lattice-ml
cd fuel-lattice-ml
python -m venv .venv                       # Python 3.12
.venv/bin/pip install -r requirements.txt  # .venv\Scripts\pip on Windows

python scripts/fetch_models.py --models rf1   # 3.97 GB from Zenodo, no account needed
.venv/bin/python cli.py predict --composition "UN0.5C0.5"
```

Training from scratch instead of downloading a binary:

```bash
.venv/bin/python cli.py build --resume            # featurization only, no MP query
.venv/bin/python cli.py train --fast --n-jobs -1  # rf1 alone
.venv/bin/python cli.py predict --composition "UN0.5C0.5"
```

**Pass `--n-jobs -1`.** It is not the default, so without it `train` fits a 600-tree forest
on a single core. On a 64-core machine rf1 takes about 45 seconds with the flag and had
not finished after 10 minutes without it. Parallelism does not change the fitted forest,
because scikit-learn draws each tree's seed from `random_state` before dispatch.

Timings depend heavily on core count, so the figures reported by `cli.py status` are rough.
On a 64-core Threadripper 3995WX featurization took 64 seconds against the roughly nine
minutes that status estimates. Only a forced re-query, `cli.py build --force`, is a
multi-hour job, and only that path needs `MP_API_KEY` in `.env`.

Predicting with `rf1` needs about 4.5 GB of free RAM, because scikit-learn copies the
memmapped tree arrays into C-owned buffers on load regardless of how the file was opened,
and `rf2` at 9.74 GB needs proportionally more.

## What this is

Four notebooks at the repository root walk through the pipeline end to end and are narrated
for learning. Every notebook imports from `engine/` rather than defining its own
hyperparameters, so hyperparameters live in exactly one place, `engine/train_models.py`'s
`model_estimators()`, pinned by `tests/test_params_parity.py`:

1. `FuelLatticeParameterDataFeaturization.ipynb`, which queries Materials Project, extracts
   symmetry, and featurizes with matminer.
2. `FuelLatticeParameterModelCreation_CrossVal.ipynb`, full-dataset fitting plus 5-fold
   cross-validation.
3. `FuelLatticeParameterModelPrediction.ipynb`, prediction against the experimental
   benchmark compositions.
4. `FuelLatticeParameterModelOptimization.ipynb`, the per-crystal-system and
   per-lattice-threshold accuracy sweep.

`exploratory/` holds five further notebooks from earlier experimentation, including a neural
network and an alternate train/test split. They are unmaintained, untested, and not
guaranteed to run against the pinned dependencies; read `exploratory/README.md` before
relying on any of them.

## Repository layout

```
engine/         the pipeline, and the single source of truth for everything
cli.py          entry point: status | build | train | evaluate | predict
server.py       FastMCP stdio server exposing predict + status as MCP tools
.mcp.json       registers the MCP server for Claude Code
.claude/skills/ lattice-predict, lattice-build, lattice-train, lattice-evaluate
Data/           committed inputs: the build seed, curated references, benchmarks
Dataset/        GENERATED and gitignored, rebuilt by `cli.py build`
Models/         model cards, hyperparameters, metrics; binaries fetched separately
Results/        the optimization study, cross-validation metrics, figures, benchmarks
exploratory/    unmaintained notebooks
scripts/        fetch_models, upload_models, write_model_manifest, basis_check,
                feature_correlations
tests/          pytest suite, including the hyperparameter-parity and seed-resume guards
```

`AVAILABILITY.md` maps every figure and table in the manuscript to the file holding its
values, and `DATA_DICTIONARY.md` describes every column of every data file.

## Using it as an agent tool

The repository is also a self-contained MCP tool. `.mcp.json` registers a
`lattice-prediction` server exposing `predict_lattice_parameter` and `artifact_status`, and
`.claude/skills/` ships four skills that a Claude Code agent can drive directly, as in
`/lattice-predict UN0.5C0.5`. Clone it standalone or as a submodule of a larger agent
project; the notebooks and the agent tooling stay in sync because they are the same
`engine/` code.

## The five models

| Key | Name | Binary | CV MAE, cubic (Å) | U(N,C) Pearson r |
|---|---|---|---|---|
| `rf1` | Lumped RF | 3.97 GB | 0.121701 | +0.9012 |
| `rf2` | Independent RF | 9.74 GB | 0.125516 | +0.9405 |
| `gbr1` | Lumped GBR (XGBoost) | 51 MB | 0.113556 | −0.0972 |
| `gbr2` | Independent GBR (HistGBR) | 23 MB | 0.151228 | −0.2696 |
| `lin` | Linear Regression | 17 KB | 1.046069 | −0.9629 |

Both columns describe **the deposited binaries**, which are the ones the manuscript reports
and the ones `scripts/fetch_models.py` downloads. All five were fitted at `random_state=42`
on 2026-07-13.

`rf1` is the headline model, and it leads on repeatability rather than on accuracy. `gbr1`
is the better cross-validated model, at 0.113556 Å against 0.121701 Å while being 77 times
smaller, but it returns a negative correlation on the U(N,C) benchmark, predicting the
lattice parameter to fall as carbon substitutes for nitrogen. A model that gets the *sign*
of the composition dependence wrong is not usable for screening, however small its mean
error. `rf2` is marginally the more accurate of the two forests but is last on
cross-validated R² over the cubic subset and weighs 9.74 GB.

**Use the deposited binaries for any number you intend to quote.** The gradient-boosting
models fit 1,800 successive rounds, each one against the previous round's residuals, so
small differences in floating-point arithmetic between machines compound from round to
round. Refitting `gbr1` on different hardware can therefore land on a measurably different
model. The random forests average 600 independently built trees, which cancels those
differences rather than accumulating them, and they refit consistently.

Both benchmarks are also extrapolation, since the fractional solid solutions they score
are absent from the training data, so the figures above are specific to the fit that
produced them.

Linear Regression is suppressed from prediction output unless `--include-baseline` is
passed, because a model this far off should not be mistaken for a usable prediction.

Two cautions apply before any of these numbers are quoted. The cubic-subset metrics are much
better than the all-systems metrics, at roughly 0.12 Å against 0.31 Å on MAE across
the 64,128
training rows, and predictions for the monoclinic, triclinic and trigonal hosts that are
thinly represented in the database are worse still, by up to 360 % on MAE. Separately, the
training labels are DFT-relaxed geometry and the benchmark values are experimental
measurements, so a benchmark MAE is not pure model error. `Results/benchmarks/basis_check.md`
quantifies that gap and explains why slope and Pearson r, which a constant offset cannot
change, are the defensible way to compare models here.

## Fetching the model binaries

The binaries are not in git, by `.gitignore`. They live in the Zenodo deposit and are
fetched with:

```bash
python scripts/fetch_models.py              # all five, about 14 GB
python scripts/fetch_models.py --models rf1 # just the headline model
```

Neither command needs an account or a token, and each verifies the download's SHA-256
against `Models/MANIFEST.json` before the file is ever unpickled, which matters because
`joblib.load` executes arbitrary code embedded in a pickle. See `Models/README.md`, which
also explains why loading uses `mmap_mode="r"` and why the binaries must stay uncompressed.

## Attribution

Trained on data from the [Materials Project](https://materialsproject.org), licensed
CC BY 4.0. Both the seed dataset and the trained models are derived works and carry the
same attribution requirement on reuse. `NOTICE` carries the funding acknowledgement, which
is quoted verbatim from the appointment terms and should not be reworded.

## Citation

Citation metadata is in `CITATION.cff`, and the Zenodo DOI above resolves to the archived
version that the manuscript's results were produced with.

## License

MIT, in LICENSE, and it covers the source code only. The Materials Project data in `Data/`
and the five models derived from it remain CC BY 4.0, so attribution travels with them
even though the code around them does not require it.
