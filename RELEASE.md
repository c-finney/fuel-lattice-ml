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

## Outstanding before the repository goes public

The ORISE Terms of Appointment require any release of information arising from the
appointment to go through the host facility's process, separately from the
question of who owns the copyright. Three items thus come from ORNL's
Innovation and Partnerships Office and are placeholders until they arrive:

- [ ] **ORNL software release number**, format `ORNL-CODE-XXXXXX`, which ORNL
      requires in README.md. Currently written as `ORNL-CODE-XXXXXX` in README.md
      and in .zenodo.json, and both occurrences must be replaced together.
- [ ] **Confirmation of the copyright line in LICENSE.** It currently reads
      `Copyright (c) 2025 Cade Finney`, on the reading that section 4 of the
      Participant Data Agreement waives ORISE and ORAU invention rights under
      35 U.S.C. 212. That reading has not been confirmed by anyone at ORNL, and
      IPO's interpretation governs rather than this file's.
- [ ] **Confirmation of the NOTICE text.** The funding acknowledgement is quoted
      verbatim from the Terms of Appointment. IPO may supply different or
      additional boilerplate, and their text wins.

The first point of contact is Denise Adorno Lopes, adornolopesd@ornl.gov, who was
the mentor of record.

## Archived version

The archived copy of record is the Zenodo deposit, DOI 10.5281/zenodo.XXXXXXX,
which contains this source tree together with the five trained model binaries.
Replace that placeholder once the deposit is published. GitHub holds the working
history; Zenodo holds the version the manuscript cites, and where the two ever
disagree the Zenodo copy is the one that was reviewed.

## What a rebuild reproduces

The pipeline runs from `Data/MP_Dataset_Original_Trimmed.csv` with no Materials
Project API key, and a rebuild on Linux under the pinned dependency versions
reproduces the following.

- Featurization returns 137,686 rows and 503 feature labels, and rewrites the
  committed label files byte-identically, so `git status` stays clean across a
  rebuild.
- The training frame comes out at 64,128 rows and 145 features, matching
  `Models/MANIFEST.json`.
- Cross-validated metrics reproduce for all five models. The largest disagreement
  with the committed `Results/metrics/ModelMetrics_CrossVal.csv` is 8.5e-4 on
  `MAE_cubic`, which changes no reported figure.
- Benchmark predictions reproduce exactly for `rf1`, `gbr2` and `lin`, and to
  within 2e-5 for `rf2`.
- `gbr1` does not reproduce. Its U(N,C) mean absolute error rebuilds as 0.0153 Å
  against a recorded 0.0397 Å, and its Pearson r as +0.8219 against a recorded
  -0.0972. Every model shares one training frame, and the four scikit-learn models
  reproduce, so this is specific to XGBoost rather than to the data. The recorded
  values were produced on Windows and scikit-learn guarantees cross-platform
  determinism where XGBoost does not, which is the likeliest cause; that has not
  been confirmed by rebuilding on Windows.
- None of the five binaries is byte-identical to the artifacts recorded in the
  2026-07 manifest, with differences from 32 bytes on `rf1` to 112 KB on `gbr1`.
  Reproducing the numbers matters here and reproducing the bytes does not, so this
  was not pursued further.

The secrets position: the Materials Project API key used during development,
prefix `csvp7B7`, was revoked on 9 July 2026 and is absent from the working tree
and from all of history. Four matches survive a naive grep and are all benign,
three quoting the prefix in prose and one the `REVOKED_KEY` sentinel that
`tests/test_params_parity.py` uses to assert it never returns. A secret scanner
will flag those four. `.env` has never been tracked, and the only `MP_API_KEY=`
occurrence in a tracked file is an error-message template in `engine/config.py`.

The 51-test suite passes under Python 3.12.10 with the pinned versions. It has
never been run on a clean clone from a fresh checkout, which is the check most
likely to catch a packaging mistake and the obvious gap in this record.

## Known problems that were not fixed

These are documented rather than resolved, and a fork inherits all of them:

- **A single benchmark correlation is not a property of a model.** Both benchmarks
  are extrapolation, since the fractional solid solutions they score are absent
  from the training data, and the boosted models move a long way under a change of
  random seed. Over seeds 42 to 46 the U(N,C) Pearson r spans +0.3764 to +0.8219
  for `gbr1`, and changes sign for `gbr2`, against +0.8919 to +0.9147 for `rf1`.
  `Results/benchmarks/seed_stability.md` has the full table. Anything quoting one
  fit's correlation, including the accompanying manuscript, is quoting a number
  with a standard deviation several times larger than the effect being described.
- The cross-validation notebook's original run reported 64,838 training rows,
  while `Models/MANIFEST.json` records 64,128 for the build behind the shipped
  models. The Materials Project database moved between the two runs and no
  database version was captured at build time. Every rebuild should record
  `MPRester.get_database_version()`, and none of them did.
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
