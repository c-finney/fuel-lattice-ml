# Data/

Committed inputs. Everything here is small, static, and required to bootstrap the
pipeline with no Materials Project API key.

## `MP_Dataset_Original_Trimmed.csv` (21 MB)

The "post-query resume point": symmetry-annotated, noble-gas-filtered Materials Project
entries, before matminer featurization. This is thus the artifact that lets
`python cli.py build --resume` skip the multi-hour Materials Project query on a fresh
clone, as described in `engine/build_dataset.py` and the root `README.md`'s Quickstart.

**Sourced from the Materials Project** (https://materialsproject.org), licensed
**CC BY 4.0**. Any redistribution of this file, or of models trained from it
(`Models/binaries/LumpedRFModel.joblib`), must retain that attribution.

## `reference_systems.json`

Curated symmetry/lattice data for 9 end-member host structures (UN, UC, CeO2, UO2,
PuO2, ThO2, ZrO2, Nd2O3, NdO2), used by `engine/mp_client.py`'s reference resolution.
This is what lets `cli.py predict` resolve a reference host for any of these systems,
by formula, by mp-id, or via automatic dominant-end-member detection, with **zero**
Materials Project API calls. Also sourced from the Materials Project (CC BY 4.0).

## `benchmarks/`

Experimental validation data with known lattice parameters (`a_true` column), used by
the prediction notebook and as regression-test fixtures:

- `UNUC.csv`, the U(N,C) system (23 rows), based on an experimental fit.
- `CeO2Nd2O3Vals.csv`, the (Ce,Nd)O2 system (7 rows).

> ### ⚠ Label-basis mismatch: read before quoting any benchmark MAE
>
> **The models are trained on DFT lattice parameters. These benchmarks are experimental
> measurements. They are not the same quantity.** A benchmark MAE is thus *not* pure
> model error, because it contains the DFT-vs-experiment discrepancy inherited through the
> training labels.
>
> `MP_Dataset_Original_Trimmed.csv` comes from the Materials Project, where **every lattice
> parameter is DFT-relaxed**. Note that MP's `theoretical: False` flag means the structure
> has been *observed*, **not** that its lattice parameters were *measured*. For
> instance `mp-1865` (UN) is flagged `theoretical: False` and still carries a
> computed a = 4.877379 Å, against an experimental 4.884 Å.
>
> The effect is real and material-dependent. For CeO2 (`mp-20194`) the DFT value exceeds the
> experimental one by **+0.057365 Å**, which accounts for essentially all of the uniform
> over-prediction every model shows on the (Ce,Nd)O2 benchmark.
>
> **No DFT→experiment correction is shipped**, because this repo cannot justify one: only 3 of
> the 9 curated hosts have an experimental value here at all, and the delta's *sign flips*
> across those 3 (UN −0.006621, UC −0.022736, CeO2 +0.057365 Å).
>
> Run **`python scripts/basis_check.py`** to regenerate `Results/benchmarks/basis_check.md`
> for the full anchor table, and for the **slope / Pearson r** metrics, which are invariant to
> a constant offset and are thus the defensible way to compare models here.

### The `ref_mp-id` column

`ref_mp-id` names the **reference host structure** whose symmetry is fed to the model
(crystal system, spacegroup, site count), so it is an input and not a label. One rule
governs it, and it is the same rule `engine/reference_resolver.py` applies when the
column is absent:

> **Use the most prevalent end-member's mp-id. For a 50/50 mix, use the more stable one's.**

Worked through for the two benchmarks:

- **`UNUC.csv`**, `U N_y C_(1-y)`. `y > 0.5` → **UN, `mp-1865`**; `y < 0.5` → **UC,
  `mp-2489`**. The `y = 0.5` row is a true tie, broken on stability: UN and UC have the
  *same* `energy_above_hull` (0.0, since each is a line compound on its own chemsys hull, so
  hull energy says nothing about which is more stable than the other), so the decision
  falls to `formation_energy_per_atom`, where **UN (−1.582 eV/atom) beats UC (−0.255)**.
  The 50/50 row thus takes `mp-1865`.
- **`CeO2Nd2O3Vals.csv`**, `Ce_x Nd_(1-x) O2`. Ce is dominant on every row (lowest Ce
  fraction is 0.6451), so **CeO2, `mp-20194`** throughout; no tie arises. NdO2 would be
  invalid as a host regardless, because Nd is 3+ and NdO2 is thus not a stable
  fluorite (see the non-host guard in `reference_resolver.py`).
