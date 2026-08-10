# Publication Checklist

Items to resolve before this repository (and its HuggingFace model repo) go public.
None of these block continued private development.

## Secrets / privacy

- [x] MP notebook key (`csvp7B7…`) revoked — user-confirmed 2026-07-09.
- [x] Verified the key never reached AERIS git history (100 commits checked, no `.env` ever tracked).
- [x] `grep -rI 'csvp7B7' .` returns nothing in the working tree (verified via `tests/test_params_parity.py::TestNotebooksImportEngine::test_no_local_paths_or_revoked_key` on every notebook, core and exploratory).
- [x] Re-verified `grep -rI 'csvp7B7' .` **and** `git log -p --all` against this repo's own
      history, 2026-07-13. Clean. The 4 history hits are all benign: three are this file's own
      prose quoting the revoked key's prefix, the fourth is the `REVOKED_KEY = "csvp7B7"` sentinel
      constant in `tests/test_params_parity.py`. `.env` has never been tracked, and the only
      `MP_API_KEY=` occurrence in a tracked file is the error-message template in
      `engine/config.py`. **A naive `grep`/secret-scanner will flag those 4 — they are expected.**
- [x] No `C:/Users/Cade` or `C:/Users/q4f` string anywhere in source (verified by the same test, and by direct `grep` across the working tree during Phase 5).

## Licensing / attribution

- [ ] **Licensing decided before the public flip.** The repo currently ships with **no
      `LICENSE` and no copyright assertion**. Public + unlicensed = all rights reserved
      by default under copyright law — nobody may legally copy, modify, or redistribute
      the code, skills, or MCP server this repository exists to distribute. That is a
      contradiction for a repo whose stated purpose is to be cloned. Resolve with ORNL
      software-release *at the flip*, not before.
- [ ] ORNL software-release approval obtained.
- [x] `NOTICE` = SULI/WDTS/ORISE funding statement, verbatim, verified pure ASCII (no
      zero-width spaces). No Advanced Fuels Campaign line, no UT-Battelle authorship
      paragraph, no `DE-AC05-00OR22725` — those belong to other projects.
- [x] `CITATION.cff`: Cade Finney, ORCID `0009-0007-6335-1536`, no `license:` key.
- [x] Funding program confirmed: DOE Office of Science, WDTS / SULI, hosted at ORNL,
      administered by ORISE.
- [x] Materials Project **CC BY 4.0** attribution present in `README.md`, `NOTICE`,
      `Data/README.md`, and `Models/LumpedRFModel/model_card.md` (the seed dataset
      *and* the trained model are both derived works).

## Repository / hosting

- [ ] Repo transferred from the personal `c-finney` GitHub namespace to `ORNL-Inria`
      (once approved); `git submodule set-url` updated in the AERIS-AgentFactory
      submodule; `.gitmodules` there committed.
- [x] HuggingFace repo created: `c-finney/fuel-lattice-ml`, **private**, model binary
      uploaded uncompressed (3,966,648,193 bytes, SHA-256 recorded in
      `Models/MANIFEST.json`).

> [!success] **RESOLVED — the 2026-07-14 staleness blocker is closed (verified 2026-08-10).**
> `rf1` was **retrained** on 2026-07-13 (`cli.py train --full`); the HF upload and the manifest
> both predated it, so a clean clone would have pulled the **OLD** `rf1` — a different model from
> the one behind every number in `Results/metrics/ModelMetrics_CrossVal.csv`, the model cards, and
> `Results/benchmarks/`. Re-upload + manifest regeneration landed in `c6b1532`.
>
> **Verified by direct measurement, not by reading the manifest** — the local binary was hashed and
> the HuggingFace copy queried via `HfApi.model_info(files_metadata=True)`:
>
> | | bytes | sha256 |
> |---|---|---|
> | local `Models/binaries/LumpedRFModel.joblib` | `3966648193` | `806e57d9…d14a1373` |
> | `Models/MANIFEST.json` → `rf1` | `3966648193` ✓ | `806e57d9…d14a1373` ✓ |
> | HuggingFace LFS record | `3966648193` ✓ | `806e57d9…d14a1373` ✓ |
>
> All three agree, so `scripts/fetch_models.py` pulls the current model and its SHA-256 gate passes.
> The superseded values were `bytes 3966646433` / `sha256 cbd71b57…0fdcab2`; they are recorded here
> only as history and **must not** be reintroduced into any card, manifest, or doc.

- [x] **Re-upload the current `rf1` to HF and regenerate `Models/MANIFEST.json`** — done in
      `c6b1532`, re-verified 2026-08-10 (table above). The HF object is named
      `LumpedRFModel.joblib`, matching the manifest `uri` after the Dependent→Lumped rename, so the
      fetch path is intact. (Do **not** upload `rf2` — 9,735,288,229 bytes and last on CV R²_cubic;
      user decision 2026-07-14.)
- [ ] HF repo flipped private → public (or transferred to an ORNL-controlled namespace
      first — same open question as the GitHub repo above).
- [ ] `scripts/fetch_models.py` verified end-to-end on a clean clone **without**
      `HF_TOKEN` set (i.e. after the HF repo is actually public). **This is the check that would
      have caught the stale-manifest blocker above — it has never been run.**

## Tests / correctness

- [x] `python -m pytest tests/ -v` — **51 passed** locally as of 2026-07-14 (includes
      `test_params_parity.py`'s notebook/hyperparameter guard, `test_seed_resume.py`'s
      seed-resume regression guard, and the reference-resolver tie-break tests).
- [ ] Re-run `python -m pytest tests/` on a genuinely clean clone (fresh `.venv`,
      no local caches) before the public flip.
- [x] End-to-end acceptance criteria from the implementation plan — see the plan
      document's §16 for the full list; verify each explicitly before publication.

## GitHub

- [ ] GitHub repo flipped private → public.
