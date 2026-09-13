# Release status

## This project is retired

Development stopped when the SULI appointment that produced it ended on 1 August
2025. The repository is frozen at the state that backs the accompanying
manuscript, and it is published so that the results in that manuscript can be
checked and reused, not as software with a future.

Concretely, that means no maintenance, no support, no releases, and no response
to issues or pull requests, none of which are monitored. The MIT license in LICENSE
disclaims warranty and liability in the usual terms, and those terms should be
read as meant rather than as boilerplate: dependency versions will drift out from
under this code, and nobody is going to fix it when they do.

Anyone who wants the work to continue should fork it, which is the intended path
and the reason the license is MIT rather than something with conditions attached.

## Correspondence

Cade Finney, cadefinney@outlook.com, ORCID 0009-0007-6335-1536. Email about the
manuscript will be answered, and email asking for a fix to the code will not be.

## Release approval

Nothing is outstanding. These are recorded because a reader may reasonably wonder
how a repository written under a DOE-sponsored appointment came to be published
under MIT:

- [x] **Manuscript review: complete.** The accompanying article was reviewed and released
      through ORNL's internal process by the ORNL co-authors; the editorial comments
      carried in the manuscript file came from that review.
- [x] **Copyright line in LICENSE: confirmed.** `Copyright (c) 2026 Cade Finney`.
      The rights the Government holds under items 1 and 3 of the Participant Data
      Agreement are recorded in NOTICE under "Government rights", and LICENSE
      points at it; those rights sit alongside the MIT grant rather than
      displacing it.
- [x] **NOTICE text: settled.** The funding acknowledgement is quoted verbatim
      from the appointment letter, which introduces it as an acknowledgement
      "such as the following" and so permits the wording as given. NOTICE also
      records the Government's rights under the Participant Data Agreement and
      the Materials Project attribution and disclaimer.

The first point of contact is Denise Adorno Lopes, adornolopesd@ornl.gov, who was
the mentor of record.

## Archived version

The archived copy of record is the Zenodo deposit, DOI 10.5281/zenodo.22727667,
which contains this source tree together with all five trained model binaries,
about 14 GB in total including the 9.74 GB `rf2`. GitHub holds the working
history; Zenodo holds the version the manuscript cites, and where the two ever
disagree the Zenodo copy is the one that was reviewed.

The deposited binaries are the artifacts fitted on **2026-07-13**, which are the
ones every number in the manuscript describes.

Each of the five `uri` fields in `Models/MANIFEST.json` reads
`zenodo://22727667/<filename>`, which `scripts/fetch_models.py` expands to the
record's file endpoint. Those URIs resolve once the deposit is published; until
then a download returns 404. The SHA-256 gate in front of the unpickle is correct
and verified against the files to be deposited.

## What the deposit reproduces

The pipeline runs from `Data/MP_Dataset_Original_Trimmed.csv` with no Materials
Project API key. The environment of record, read from the virtualenv that fitted
the deposited binaries on 2026-07-13, is Python 3.12.10 with scikit-learn 1.9.0,
XGBoost 3.3.0, NumPy 2.5.1, pandas 2.3.3, joblib 1.5.3, pymatgen 2026.5.4 /
pymatgen-core 2026.5.18 and matminer 0.10.1.

The inputs rebuild byte-identically on any machine:

- Featurization returns 137,686 rows and 503 feature labels and rewrites the
  committed label files byte-identically, so `git status` stays clean.
- `Dataset/MP_Dataset_Featurized.csv` comes out at SHA-256
  `8bff88c700d2ce98d229de0752975988ba2d5f2f2562a64395f8d47af074ab01`.
- `Dataset/Training_Dataset.csv` comes out at 64,128 rows and 145 features,
  SHA-256 `5903296149f451383dfe32e266fb1f28966e4411eaf946f221f839e7dacfcb2a`,
  matching `Models/MANIFEST.json`.

The deposited binaries reproduce every model number reported. Scored against both
solid-solution benchmarks they return the manuscript's benchmark table in full,
all five models, on MAE, slope and Pearson r, including `gbr1` at MAE 0.039714 Å,
slope -0.164, Pearson r -0.0972. `Results/benchmarks/basis_check.md` is that
table, regenerated from them, and `Results/benchmarks/UNUC/` and
`Results/benchmarks/CeO2Nd2O3/` hold the per-composition predictions behind it.

`Models/MANIFEST.json` carries the size and SHA-256 digest of each binary,
`scripts/fetch_models.py` verifies a download against it before the file is
unpickled, and `AVAILABILITY.md` maps every figure and table in the manuscript to
a file in the deposit.

The secrets position: the Materials Project API key used during development,
prefix `csvp7B7`, was revoked on 9 July 2026 and is absent from the working tree
and from all of history. Four matches survive a naive grep and are all benign,
three quoting the prefix in prose and one the `REVOKED_KEY` sentinel that
`tests/test_params_parity.py` uses to assert it never returns. A secret scanner
will flag those four. `.env` has never been tracked, and the only `MP_API_KEY=`
occurrence in a tracked file is an error-message template in `engine/config.py`.

The 69-test suite passes under Python 3.12.10 with the pinned versions, and it
passes from a clean clone into an empty directory with a fresh virtualenv built
from `requirements.txt` alone, with no model binaries present.

That clean install resolves to XGBoost 3.4.1 and NumPy 2.5.3 rather than the
3.3.0 and 2.5.1 the binaries were fitted under, because `requirements.txt`
lower-bounds XGBoost rather than pinning it. The deposited `gbr1` was loaded and
scored under 3.4.1 and returns MAE 0.039714 Å and MSE 0.003802 Å² on U(N,C),
matching the values in `Results/benchmarks/basis_check.md` exactly, so the
artifact is readable across that gap. A future XGBoost release could change
that; `Models/MANIFEST.json` records the version the pickles were written under.

## Known problems that were not fixed

These are documented rather than resolved, and a fork inherits all of them:

- **The gradient-boosting models have to be taken from the deposited binaries
  rather than refitted.** Boosting fits 1,800 successive rounds, each against the
  previous round's residuals, so differences in floating-point arithmetic between
  machines compound from round to round instead of cancelling. Refitting `gbr1`
  elsewhere can produce a measurably different model. A random forest averages 600
  independently built trees and is not exposed to this, which is part of why the
  headline model is `rf1`.
- **Both benchmarks are extrapolation**, since the fractional solid solutions they
  score are absent from the training data, and they are 23 and 7 compositions
  respectively. Figures from them are specific to the fit that produced them, and
  everything here is fitted at `random_state=42`.
- The cross-validation notebook's original run reported 64,838 training rows,
  while `Models/MANIFEST.json` records 64,128 for the build behind the deposited
  models. The Materials Project database moved between the two runs. The build
  behind the deposited models is pinned by `queried_at`,
  2026-06-25T15:03:29.794248+00:00, which identifies the snapshot; `db_version`
  is left null because the query date is what fixes the data and the earlier
  64,838-row run is not the one being deposited.
- `Results/archive/CrystalSystemRandomForestRegressorOptimizationStudy_Full.csv`
  disagrees with the `_Final.csv` that superseded it on MSE, MAE and R2 while
  agreeing on every row's dataset size. Two runs of what should have been one
  configuration; the discrepancy was never explained. See Results/README.md.
- The model binaries are joblib pickles, which execute arbitrary code on load.
  `scripts/fetch_models.py` verifies SHA-256 against the manifest before
  unpickling, and that is a mitigation rather than a fix. Serializing to skops or
  ONNX would remove the risk and was never done.
- Predictions for non-cubic hosts are substantially worse than the headline cubic
  numbers suggest, by up to 360 % on MAE for the least represented systems.
