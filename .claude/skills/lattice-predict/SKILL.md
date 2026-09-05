---
name: lattice-predict
description: >
  Predict the lattice parameter(s) of a solid-solution composition (e.g. "UN0.5C0.5")
  using the trained Materials-Project ML model shipped in this repository. Auto-offers
  to build/train artifacts if they are missing. Use when the user asks for a predicted
  lattice parameter, unit cell size, or invokes /lattice-predict.
---

Predict lattice parameters using the trained model in `Models/binaries/` via `cli.py`
at the repository root (`engine/predict.py` is the source of truth; this skill
orchestrates the prereq check, the build/train confirmation, and output formatting).

## Usage

- `/lattice-predict UN0.5C0.5`: single composition (natural-language friendly).
- `/lattice-predict "U1 N0.5 C0.5"`: spaced formula also accepted.
- `/lattice-predict Data/benchmarks/UNUC.csv`: batch CSV
  (`composition`[,`ref_mp-id`,`y`,`a_true`]).
- `/lattice-predict Ce0.8343Nd0.1657O2 reference=mp-20194`: force the reference structure.
- `/lattice-predict UC0.7N0.3 models=rf1,gbr1`: restrict to specific models.

## Procedure

1. **Parse the argument** into either a composition string or a CSV path. Pull optional
   `reference=` and `models=` tokens.

2. **Check prerequisites** (fast, no heavy work):
   ```
   python cli.py status --json
   ```
   Prediction needs `Models/feature_labels/ML_FeatureLabels.joblib` **and** at least one
   non-LR trained model. It does **not** need the full featurized dataset, and it does
   **not** need `MP_API_KEY` for any composition whose reference host is one of the 9
   curated end-members in `Data/reference_systems.json` (UN, UC, CeO2, UO2, PuO2, ThO2,
   ZrO2, Nd2O3, NdO2), by formula, mp-id, or automatic dominant-end-member resolution.

3. **If prerequisites are missing → confirm before any heavy work (MANDATORY):**
   - State exactly what's missing and the **build** ETA from the status output. With the
     committed seed dataset present, a normal build is ~9 minutes, and only a forced
     re-query (`--force`) is a multi-hour job. Say which applies.
   - Ask the user **yes/no**. Do **not** start building without a "yes".
   - On **yes**, run in the **background** (`run_in_background: true`), narrating progress:
     - If the featurized dataset is missing: run the build first (`cli.py build --resume`),
       then a fast train.
     - Fast train (Lumped RF only, the fastest option and the headline model):
       `cli.py train --fast`.
     - When the background job finishes, continue to step 4.
   - On **no**, stop and tell the user they can run `/lattice-build` then `/lattice-train`
     later.

4. **Predict:**
   ```
   python cli.py predict --composition "<comp>" [--reference <ref>] [--models <list>] --json
   ```
   (Use `--csv <path>` instead of `--composition` for batch mode. Add `--out-dir` if
   the user wants a saved copy of the result.)

5. **Handle the engine status field:**
   - `needs_reference` → the solid solution's host structure is ambiguous (e.g. >2 mixed
     components, or the dominant end-member isn't a stable host). Show the candidate
     end-members/mp-ids the engine returned and ask the user to pick one, then re-run with
     `--reference`.
   - `needs_build` → return to step 3.
   - `ok` → format the answer (step 6).

6. **Report:**
   - **Headline:** the **Lumped RF** value (state the model name). For a **cubic**
     host report a single lattice parameter `a` (Å); for **non-cubic** report `a`, `b`,
     `c` separately and add the non-cubic accuracy caution.
   - **Table:** all available models **except Linear Regression**, one row each
     (a, plus b/c if non-cubic; plus error/MAE columns if `a_true` was supplied).
   - **Provenance line:** the resolved reference end-member, its `mp-id`, crystal system,
     spacegroup, the tie-break basis (`energy_above_hull` if it was a stoichiometric tie),
     and whether it came from the curated table, a runtime cache, or a live Materials
     Project lookup, so the user can sanity-check the host.
   - Surface any out-of-domain warning the engine emits (element absent from training).

## Numeric-reporting note

Every reported value is an ML **prediction**, not a measured/ground-truth value. Label
it as such, give units (Å), name the model, and cite the resolved reference structure.
Never present a prediction as a verified value; if the engine flags out-of-domain, say
so explicitly.
