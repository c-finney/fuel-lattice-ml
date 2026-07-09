"""
build_dataset.py — Notebook 1 (trimmed, resumable).

Pipeline:
  1. Query MP (skip if the trimmed dataset already exists & --resume)
  2. Noble-gas filter
  3. extract_symmetry per doc -> save Original_Trimmed (resume point)
  4. <= BUILD_THRESH filter
  5. OHE crystal_system + spacegroup_num
  6. Featurize compositions (composition_reduced)
  7. Save Featurized CSV + FeatureLabels joblibs
  8. mark_stage

Hidden --limit N (or MP_LIMIT env) caps MP query for smoke-testing.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import joblib
import pandas as pd

from engine import artifacts, config, featurize, mp_client

NOBLE_GASES = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"})


def build(
    thresh: float = config.BUILD_THRESH,
    resume: bool = True,
    force: bool = False,
    progress: Callable = print,
    limit: int | None = None,
) -> dict:
    """
    Build the featurized Materials Project dataset.

    Parameters
    ----------
    thresh  : lattice-parameter threshold in Å for the build step (default 20)
    resume  : if True and the trimmed dataset already exists, skip the MP
              query stage (see config.dataset_original())
    force   : if True, ignore resume and always re-query MP
    progress: callable for status messages (default print)
    limit   : if set, cap the MP query to this many docs (smoke-test only)

    Returns
    -------
    dict with keys: rows, paths
    """
    # ------------------------------------------------------------------ #
    # Stage 1 — MP query + per-doc symmetry extraction                    #
    #                                                                      #
    # The resume signal is FILE EXISTENCE, not a manifest stage. A fresh   #
    # clone ships the trimmed dataset as a committed seed                 #
    # (Data/MP_Dataset_Original_Trimmed.csv) but has no build manifest —  #
    # gating on "query in stages_done" as well would force a multi-hour   #
    # re-query on every first run, defeating the entire point of shipping #
    # the seed.                                                           #
    # ------------------------------------------------------------------ #
    if resume and not force and config.dataset_original().exists():
        progress("[build] Stage 1/3: query — SKIPPED (trimmed dataset present)")
        df_orig = pd.read_csv(config.dataset_original(), low_memory=False)
    else:
        progress("[build] Stage 1/3: querying Materials Project…")
        try:
            docs = mp_client.search_summary(
                fields=config.QUERY_FIELDS,
                deprecated=False,
            )
        except Exception as exc:
            raise RuntimeError(f"MP query failed: {exc}") from exc

        progress(f"[build]   Got {len(docs)} docs from MP")

        # Apply MP_LIMIT / --limit
        cap = limit or int(os.environ.get("MP_LIMIT", 0))
        if cap:
            docs = docs[:cap]
            progress(f"[build]   Capped to {cap} docs (smoke-test mode)")

        # Noble-gas filter
        filtered = [d for d in docs if not (NOBLE_GASES & {el.symbol for el in d.elements})]
        progress(f"[build]   After noble-gas filter: {len(filtered)} docs")

        # Per-doc extraction
        rows = []
        errors = []
        for i, doc in enumerate(filtered):
            if i % 500 == 0:
                progress(f"[build]   Extracting symmetry {i}/{len(filtered)}…")
            try:
                sym = mp_client.extract_symmetry(doc.structure, doc.nsites)
                row = {
                    "material_id":              doc.material_id,
                    "nsites":                   doc.nsites,        # raw doc.nsites — see mp_client.py
                    "nelements":                doc.nelements,     # dedup only, not a model feature
                    "composition_reduced":      str(doc.composition_reduced),
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    **sym,
                }
                rows.append(row)
            except Exception as exc:
                errors.append(f"{doc.material_id}: {exc}")

        if errors:
            progress(f"[build]   Extraction errors ({len(errors)}): {errors[:5]}")

        df_orig = pd.DataFrame(rows)
        # Write to the GENERATED location — never overwrite the committed
        # seed in Data/. config.dataset_original() will prefer this file
        # over the seed on the next run.
        config.DATASETS.mkdir(parents=True, exist_ok=True)
        gen_path = config.DATASETS / "MP_Dataset_Original_Trimmed.csv"
        df_orig.to_csv(gen_path, index=False)
        progress(f"[build]   Saved Original_Trimmed: {len(df_orig)} rows")

        artifacts.mark_stage("query", rows=len(df_orig), errors=len(errors))

    # ------------------------------------------------------------------ #
    # Stage 2 — Lattice-parameter filter + OHE                            #
    # ------------------------------------------------------------------ #
    progress(f"[build] Stage 2/3: lattice filter (<= {thresh} Å) + OHE…")
    df = df_orig.copy()
    before = len(df)
    df = df[(df[["a", "b", "c"]] <= thresh).all(axis=1)].copy()
    progress(f"[build]   {before} -> {len(df)} rows after {thresh} Å filter")

    # OHE — fixed label sets from config, never re-derived from the data
    df_cs = featurize.ohe_crystal_system(df)
    df_sg = featurize.ohe_spacegroup(df)
    df = pd.concat([df, df_cs, df_sg], axis=1)

    # Composition objects (from composition_reduced — build path; predict
    # deliberately featurizes the FULL composition string instead, see
    # predict.py)
    from pymatgen.core import Composition
    df["comp_obj"] = [Composition(x) for x in df["composition_reduced"]]

    # ------------------------------------------------------------------ #
    # Stage 3 — Matminer featurization                                    #
    # ------------------------------------------------------------------ #
    progress("[build] Stage 3/3: matminer featurization…")
    fzer = featurize.make_featurizer()
    feature_labels = featurize.base_feature_labels(fzer)

    df = featurize.featurize_compositions(df, "comp_obj", fzer)
    progress(f"[build]   Featurization complete: {len(df)} rows")

    # Drop comp_obj — it was transient; the MP `structure` object was never
    # persisted either (see config.QUERY_FIELDS).
    df = df.drop(columns=["comp_obj"], errors="ignore")

    # Save featurized dataset
    config.DATASETS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_FEATURIZED, index=False)
    progress(f"[build]   Saved Featurized CSV: {len(df)} rows")

    # Save feature labels (+ JSON mirrors so the label contract is human
    # readable without unpickling — see Models/README.md)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_labels, config.FEATURELABELS)
    _write_json_mirror(config.FEATURELABELS, feature_labels)
    joblib.dump(config.CS_LABELS, config.FEATURE_DIR / "cs_FeatureLabels.joblib")
    joblib.dump(config.SG_LABELS, config.FEATURE_DIR / "sg_FeatureLabels.joblib")
    progress(f"[build]   Saved FeatureLabels ({len(feature_labels)} labels)")

    assert "nelements" not in feature_labels, "BUG: nelements must not be in feature labels"

    artifacts.mark_stage(
        "featurize",
        rows=len(df),
        feature_count=len(feature_labels),
        thresh=thresh,
    )

    return {
        "rows": len(df),
        "paths": {
            "original": str(config.dataset_original()),
            "featurized": str(config.DATASET_FEATURIZED),
            "feature_labels": str(config.FEATURELABELS),
        },
    }


def _write_json_mirror(joblib_path, obj) -> None:
    """Write a human-readable JSON mirror next to a committed .joblib label list."""
    import json
    joblib_path.with_suffix(".json").write_text(
        json.dumps(obj, indent=2), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the featurized MP dataset (Notebook 1)."
    )
    parser.add_argument("--thresh", type=float, default=config.BUILD_THRESH,
                        help=f"Lattice-parameter threshold in Å (default {config.BUILD_THRESH})")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip MP query if the trimmed dataset already exists (default True)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore resume, always re-query Materials Project")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap MP query to N docs (smoke-test only; omit for production)")
    args = parser.parse_args(argv)

    result = build(
        thresh=args.thresh,
        resume=args.resume,
        force=args.force,
        limit=args.limit,
    )
    print(f"[build] Done: {result['rows']} rows in featurized dataset.")
    for k, v in result["paths"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
