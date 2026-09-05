---
name: lattice-build
description: >
  Build the featurized Materials-Project training dataset for lattice-parameter
  prediction. Normally fast (~9 minutes) because the post-query resume point ships
  committed in Data/; only a forced re-query (--force) is a multi-hour job.
  Use when the user asks to build, rebuild, or refresh the lattice-parameter training
  dataset, or invokes /lattice-build.
---

Build the featurized dataset used to train the lattice-parameter models, via `cli.py`
at the repository root (the engine in `engine/` is the source of truth; this skill is a
thin, narrated wrapper).

## Usage

- `/lattice-build`: build if not already present (resumes from the committed seed).
- `/lattice-build --force`: ignore the seed and re-query Materials Project from scratch.

## Procedure

1. **Check status first** (fast, no heavy work):
   ```
   python cli.py status --json
   ```
   If `Data/MP_Dataset_Original_Trimmed.csv` is present (it is, in a normal clone),
   `build --resume` skips the Materials Project query entirely and only re-runs
   featurization (~9 minutes). State this ETA plainly, since it is **not** the old
   multi-hour figure.

2. **Only warn about a multi-hour job when `--force` is requested**, or when the
   trimmed dataset is genuinely absent (`build.ok == false` in the status output with
   `MP_API_KEY` missing). In that case: state that a full rebuild is a **one-time,
   multi-hour** job (full Materials Project download + ~50k `SpacegroupAnalyzer` runs +
   matminer featurization of the compositions), and ask the user **yes/no** before
   starting. Do not start without an explicit "yes".

3. **Run**, backgrounding if it's a `--force` rebuild:
   ```
   python cli.py build --resume
   #   or, for a forced rebuild:
   python cli.py build --force
   #   optional: --thresh 20 overrides the build Å threshold
   ```
   (Activate the repo's `.venv` first if `python` doesn't already resolve to it.)
   The job is resumable, so if the query stage already completed a re-run reuses the
   trimmed dataset and only re-featurizes.

4. **On completion**, report the final row count and the artifacts written
   (`Dataset/MP_Dataset_Featurized.csv`, `Models/feature_labels/*`), and that the next
   step is `/lattice-train` (default fast = Lumped RF only). The build does **not**
   train any models.

## Notes

- Produces only the dataset + feature-label artifacts; it never trains models.
- Generated artifacts land in `Dataset/` and `Models/binaries/` (both git-ignored,
  regenerable). The committed seed lives in `Data/` and is never overwritten by a
  normal `--resume` build.
