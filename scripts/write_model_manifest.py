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


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for mod_name in ["sklearn", "xgboost", "numpy", "pandas", "joblib", "pymatgen", "matminer"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = None
    return versions


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

    manifest = {
        "schema_version": 1,
        "provenance": {
            "materials_project": {
                "client": "mp-api",
                "queried_at": query_stage.get("completed_at"),
                "db_version": None,  # not captured during the original build; see Models/README.md
            },
            "packages": package_versions(),
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
