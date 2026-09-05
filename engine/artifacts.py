"""
artifacts.py — MANIFEST read/write + prereq checks.

BUILD_MANIFEST tracks which pipeline stages are complete so the build is
resumable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from engine import config


# ---------------------------------------------------------------------------
# MANIFEST helpers
# ---------------------------------------------------------------------------

def read_manifest() -> dict:
    """Return the BUILD_MANIFEST dict; empty dict if file doesn't exist."""
    if not config.BUILD_MANIFEST.exists():
        return {}
    try:
        return json.loads(config.BUILD_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_manifest(d: dict) -> None:
    """Atomically write *d* to BUILD_MANIFEST."""
    config.BUILD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.BUILD_MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    tmp.replace(config.BUILD_MANIFEST)


# Distribution name per import name, where they differ. pymatgen ships as a
# namespace package whose top-level module carries no __version__.
_DIST_NAME = {"sklearn": "scikit-learn", "pymatgen": "pymatgen-core"}

_TRACKED = ["sklearn", "xgboost", "numpy", "pandas", "joblib", "pymatgen", "matminer"]


def package_versions() -> dict:
    """
    Versions of the packages that determine a fitted model's bytes.

    Recorded at fit time by train(), because Models/MANIFEST.json's own
    provenance block is written by scripts/write_model_manifest.py from whatever
    interpreter happens to run it. Those are not the same environment: the
    manifest committed in c6b1532 was regenerated weeks after the models it
    describes were fitted, so its version numbers described the machine that ran
    the script rather than the one that did the training.
    """
    import sys
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for name in _TRACKED:
        v = None
        try:
            v = getattr(__import__(name), "__version__", None)
        except ImportError:
            out[name] = None
            continue
        if not v:
            try:
                v = version(_DIST_NAME.get(name, name))
            except PackageNotFoundError:
                v = "unknown"
        out[name] = v
    return out


def mark_stage(stage: str, **meta: Any) -> None:
    """
    Record a completed stage in BUILD_MANIFEST.

    *stage* is one of:
      "query"        — MP download + Original_Trimmed saved
      "featurize"    — featurized CSV + FeatureLabels saved
      "train:<key>"  — a specific model trained (e.g. "train:rf1")
    """
    manifest = read_manifest()
    manifest.setdefault("stages", {})[stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    write_manifest(manifest)


# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return list of REPORTABLE model keys whose files exist on disk."""
    return [k for k in config.REPORTABLE if config.model_exists(k)]


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

# The committed seed dataset (Data/MP_Dataset_Original_Trimmed.csv) means a
# normal build skips the Materials Project query entirely — see
# build_dataset.py. The multi-hour figure only applies to a forced re-query
# (`cli.py build --force`), which re-downloads the full MP snapshot.
_BUILD_ETA_RESUME = "~9 minutes (featurization only; seed dataset already present)"
_BUILD_ETA_FORCE  = "tens of minutes to ~2 hours (full MP re-download + ~50k SpacegroupAnalyzer + matminer featurization)"
_TRAIN_ETA  = ("~10 minutes (rf1 fast mode) or up to ~60 minutes (full suite), "
               "and much longer without --n-jobs -1, which is not the default")


def check_prereqs(stage: str) -> dict:
    """
    Return a structured prereq status for *stage*.

    stage in {"build", "train", "predict", "evaluate"}

    Returns::
        {
          "ok": bool,
          "missing": [str],          # human-readable missing items
          "warnings": [str],         # non-blocking notes
          "build_eta": str,
          "train_eta": str,
          "available_models": [str],
        }
    """
    missing: list[str] = []
    warnings: list[str] = []
    av = available_models()

    if stage == "build":
        # With the committed seed present, `build --resume` never touches
        # the network, so a missing MP_API_KEY is not a blocker. Only a
        # forced re-query (`--force`) needs it.
        if not config.dataset_original().exists():
            try:
                config.load_mp_key()
            except RuntimeError as exc:
                missing.append(str(exc))

    elif stage == "train":
        if not config.DATASET_FEATURIZED.exists():
            missing.append("MP_Dataset_Featurized.csv (run /lattice-build first)")
        if not config.FEATURELABELS.exists():
            missing.append("FeatureLabels.joblib (run /lattice-build first)")

    elif stage in ("predict", "evaluate"):
        if not config.ML_FEATURELABELS.exists():
            missing.append("ML_FeatureLabels.joblib (run /lattice-train first)")
        if not av:
            missing.append("at least one trained model (run /lattice-train first)")
        # A missing MP_API_KEY does NOT block predict/evaluate: reference
        # resolution for any of the 9 curated end-members (or an explicit
        # --reference mp-id already in that table) works with zero network
        # calls. It only matters for an out-of-domain host, which surfaces
        # its own "needs_reference" error at predict time — not here.
        try:
            config.load_mp_key()
        except RuntimeError:
            warnings.append(
                "MP_API_KEY not set — predictions are limited to curated or "
                "cached reference hosts (see Data/reference_systems.json)."
            )

    build_eta = _BUILD_ETA_RESUME if config.dataset_original().exists() else _BUILD_ETA_FORCE

    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "warnings": warnings,
        "build_eta": build_eta,
        "train_eta": _TRAIN_ETA,
        "available_models": av,
    }
