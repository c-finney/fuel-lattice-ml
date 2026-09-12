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

- `UNUC.csv`, the U(N,C) system (23 rows). Nineteen rows, at y = 0.05 to 0.95 in steps of
  0.05, are sampled from the quadratic fit to the literature values that Figure 1 of the
  manuscript plots. Recovering that fit from those rows gives

      a(y) = -0.0139749 y^2 - 0.0573515 y + 4.9612879   (y = N fraction, a in angstrom)

  to within 1.9e-4 angstrom, so the curve is fully reproducible from this file. The
  remaining four rows are measured values rather than fit samples: the UN and UC
  end-members from Wyckoff, and two U(N,C) compositions at y = 0.98412 and y = 0.94510.
  The individual literature measurements behind the fit are in the cited sources.
- `CeO2Nd2O3Vals.csv`, the (Ce,Nd)O2 system (7 rows).

## `xrd/`

Laboratory X-ray diffraction for two UN MiniFuel specimens, with the final Rietveld
refinement of each. Cu anode, coupled TwoTheta/Theta, 20 to 92 degrees 2-theta in 0.02
degree steps at 5 s per step, 3,561 points per pattern.

| Folder | Lab specimen id | Refined a (Å) | wR |
|---|---|---|---|
| `UN-116/` | 35-p-24-093 | 4.88919 | 9.14 % |
| `UN-143/` | 35-P-24-167 | 4.89493 | 9.90 % |

Both refinements start from the same UN structure model, `UN_Wyckoff.cif`, which is COD
entry 9008757 — the same entry discussed in `Results/benchmarks/basis_check.md`, where its
missing measurement temperature and missing uncertainty are set out.

Each folder holds the raw pattern, the refinement plot, the starting model and the final
refinement only. The four earlier refinement stages (background, lattice parameter and
sample displacement, Uiso, domain size and microstrain) and GSAS-II's `.bak` autosaves are
not included.

| File | What it is |
|---|---|
| `*_exported.txt` | Raw diffractogram. One header line carrying the specimen id, anode and scan type, then 3,561 rows of 2-theta and intensity, space separated. |
| `UN-*.png` | Plot of the final refinement: observed, calculated, background and difference. |
| `UN_Wyckoff.cif` | Starting structure model, COD 9008757. Identical in both folders. |
| `UN-*_Final_4.gpx` | GSAS-II project file for the final refinement. Opens in GSAS-II. |
| `UN-*_Final_4.lst` | Refinement log: refined cell, agreement factors, parameter table. |
| `UN-*_Final_4_Histogram.csv` | Point-level fit, one row per measured point. |
| `UN-*_Final_4_ReflectionList*.csv` | Reflection table for the fitted phase. |

> ### Label-basis mismatch: read before quoting any benchmark MAE
>
> The models are trained on DFT lattice parameters and these benchmarks are experimental
> measurements, which are not the same quantity. A benchmark MAE is therefore not pure
> model error, because it contains the DFT-vs-experiment discrepancy inherited through the
> training labels.
>
> `MP_Dataset_Original_Trimmed.csv` comes from the Materials Project, where every lattice
> parameter is DFT-relaxed. MP's `theoretical: False` flag means the structure has been
> observed, not that its lattice parameters were measured: `mp-1865` (UN) is flagged
> `theoretical: False` and still carries a computed a = 4.877379 Å against an experimental
> 4.884 Å.
>
> The effect is material-dependent. For CeO2 (`mp-20194`) the DFT value exceeds the
> experimental one by +0.057365 Å, which accounts for most of the uniform over-prediction
> every model shows on the (Ce,Nd)O2 benchmark.
>
> No DFT→experiment correction is shipped, because this repository cannot justify one: only
> 3 of the 9 curated hosts have an experimental value here at all, and the sign of the delta
> flips across those 3 (UN −0.006621, UC −0.022736, CeO2 +0.057365 Å).
>
> Run `python scripts/basis_check.py` to regenerate `Results/benchmarks/basis_check.md` for
> the full anchor table and for the slope and Pearson r metrics, which are invariant to a
> constant offset and are the way to compare models here.

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
