"""
write_model_manifest.py — Regenerate Models/MANIFEST.json from the live
environment and the current contents of Models/binaries/.

Usage:
  python scripts/write_model_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine import config  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


# Distribution name per import name, where they differ. pymatgen ships as a
# namespace package with no __version__ attribute on the top-level module, so
# reading it by import returned "unknown" in every manifest written before this.
_DIST_NAME = {"sklearn": "scikit-learn", "pymatgen": "pymatgen-core"}


def package_versions() -> dict:
    """Versions in THIS interpreter. See resolve_packages() for why that differs."""
    from engine import artifacts
    return artifacts.package_versions()


def resolve_packages(stages: dict) -> tuple[dict, str]:
    """
    Prefer the environment recorded when the models were fitted.

    This script reads whatever interpreter runs it, which is not necessarily the
    one that did the training. The manifest committed in c6b1532 was regenerated
    weeks after its models were fitted, so its `packages` block described the
    machine that ran this script. That mattered: rebuilding gbr1 from the same
    seed produced a different binary and different benchmark numbers, and the
    recorded xgboost version could not be used to rule version drift in or out,
    because it was never a record of the training environment in the first place.

    train() now stamps versions into each train:<key> stage. When every trained
    model carries the same stamp, that is what gets recorded. Disagreement
    between models means they were fitted under different environments, which is
    worth surfacing rather than flattening.
    """
    stamps = {k: v["packages"] for k, v in stages.items()
              if k.startswith("train:") and isinstance(v, dict) and v.get("packages")}
    if not stamps:
        return package_versions(), "live interpreter (models predate version stamping)"
    distinct = {json.dumps(v, sort_keys=True) for v in stamps.values()}
    if len(distinct) == 1:
        return next(iter(stamps.values())), "recorded at fit time"
    return (
        {"MIXED": {k.split(":", 1)[1]: v for k, v in stamps.items()}},
        "models were fitted under DIFFERENT environments",
    )


def build_manifest() -> dict:
    build_manifest = {}
    if config.BUILD_MANIFEST.exists():
        try:
            build_manifest = json.loads(config.BUILD_MANIFEST.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    stages = build_manifest.get("stages", {})
    query_stage = stages.get("query", {})
    train_rf1 = stages.get("train:rf1", {})
    _packages, _packages_source = resolve_packages(stages)

    manifest = {
        "schema_version": 1,
        "provenance": {
            "materials_project": {
                "client": "mp-api",
                "queried_at": query_stage.get("completed_at"),
                "db_version": None,  # not captured during the original build; see Models/README.md
            },
            "packages": _packages,
            "packages_source": _packages_source,
            "training_rows": train_rf1.get("rows"),
            "n_features": train_rf1.get("features"),
            "build_thresh_angstrom": config.BUILD_THRESH,
            "train_thresh_angstrom": config.TRAIN_THRESH,
        },
        "models": {},
    }

    for key, (fname, _label) in config.MODEL_FILES.items():
        path = config.MODELS_DIR / fname
        if path.exists():
            manifest["models"][key] = {
                "file": fname,
                "compressed": False,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "uri": None,  # populated by scripts/upload_models.py after upload
                "status": "trained",
            }
        else:
            manifest["models"][key] = {"status": "not_trained"}

    return manifest


def main():
    manifest = build_manifest()
    config.MODEL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # Preserve any existing `uri` fields when regenerating (e.g. after an HF upload)
    if config.MODEL_MANIFEST.exists():
        try:
            existing = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
            for key, entry in manifest["models"].items():
                old = existing.get("models", {}).get(key, {})
                if entry.get("status") == "trained" and old.get("uri") and old.get("sha256") == entry.get("sha256"):
                    entry["uri"] = old["uri"]
        except (json.JSONDecodeError, OSError):
            pass

    config.MODEL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {config.MODEL_MANIFEST}")
    for key, entry in manifest["models"].items():
        print(f"  {key}: {entry.get('status')}"
              + (f" ({entry['bytes']:,} bytes)" if entry.get("bytes") else ""))


if __name__ == "__main__":
    main()
