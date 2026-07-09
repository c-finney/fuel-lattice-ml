# Data/

Committed inputs — everything here is small, static, and required to bootstrap the
pipeline with no Materials Project API key.

## `MP_Dataset_Original_Trimmed.csv` (21 MB)

The "post-query resume point": symmetry-annotated, noble-gas-filtered Materials Project
entries, before matminer featurization. This is the artifact that lets
`python cli.py build --resume` skip the multi-hour Materials Project query on a fresh
clone — see `engine/build_dataset.py` and the root `README.md`'s Quickstart.

**Sourced from the Materials Project** (https://materialsproject.org), licensed
**CC BY 4.0**. Any redistribution of this file, or of models trained from it
(`Models/binaries/DependentRFModel.joblib`), must retain that attribution.

## `reference_systems.json`

Curated symmetry/lattice data for 9 end-member host structures (UN, UC, CeO2, UO2,
PuO2, ThO2, ZrO2, Nd2O3, NdO2), used by `engine/mp_client.py`'s reference resolution.
This is what lets `cli.py predict` resolve a reference host for any of these systems —
by formula, by mp-id, or via automatic dominant-end-member detection — with **zero**
Materials Project API calls. Also sourced from the Materials Project (CC BY 4.0).

## `benchmarks/`

Experimental validation data with known lattice parameters (`a_true` column), used by
the prediction notebook and as regression-test fixtures:

- `CompoundsToPredict.csv` — the U(N,C) system, based on an experimental fit.
- `CeO2Nd2O3Vals.csv` — the (Ce,Nd)O2 system.
