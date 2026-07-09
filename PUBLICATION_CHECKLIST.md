# Publication Checklist

Items to resolve before this repository (and its HuggingFace model repo) go public.
None of these block continued private development.

## Secrets / privacy

- [x] MP notebook key (`csvp7B7…`) revoked — user-confirmed 2026-07-09.
- [x] Verified the key never reached AERIS git history (100 commits checked, no `.env` ever tracked).
- [x] `grep -rI 'csvp7B7' .` returns nothing in the working tree (verified via `tests/test_params_parity.py::TestNotebooksImportEngine::test_no_local_paths_or_revoked_key` on every notebook, core and exploratory).
- [ ] Re-verify `grep -rI 'csvp7B7' .` **and** `git log -p` after `git init` + first commit (this repo's own history, not AERIS's — should be clean since it's a fresh `git init`, but confirm).
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
      `Data/README.md`, and `Models/DependentRFModel/model_card.md` (the seed dataset
      *and* the trained model are both derived works).

## Repository / hosting

- [ ] Repo transferred from the personal `c-finney` GitHub namespace to `ORNL-Inria`
      (once approved); `git submodule set-url` updated in the AERIS-AgentFactory
      submodule; `.gitmodules` there committed.
- [x] HuggingFace repo created: `c-finney/fuel-lattice-ml`, **private**, model binary
      uploaded uncompressed (3,966,646,433 bytes, SHA-256 recorded in
      `Models/MANIFEST.json`).
- [ ] HF repo flipped private → public (or transferred to an ORNL-controlled namespace
      first — same open question as the GitHub repo above).
- [ ] `scripts/fetch_models.py` verified end-to-end on a clean clone **without**
      `HF_TOKEN` set (i.e. after the HF repo is actually public).

## Tests / correctness

- [x] `python -m pytest tests/ -v` — **50 passed** locally (includes
      `test_params_parity.py`'s notebook/hyperparameter guard and
      `test_seed_resume.py`'s seed-resume regression guard).
- [ ] Re-run `python -m pytest tests/` on a genuinely clean clone (fresh `.venv`,
      no local caches) before the public flip.
- [x] End-to-end acceptance criteria from the implementation plan — see the plan
      document's §16 for the full list; verify each explicitly before publication.

## GitHub

- [ ] GitHub repo flipped private → public.
