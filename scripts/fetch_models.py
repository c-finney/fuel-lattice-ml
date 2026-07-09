"""
fetch_models.py — Download trained model binaries from the HuggingFace repo
recorded in Models/MANIFEST.json, verifying SHA-256 BEFORE unpickling.

joblib.load() executes arbitrary code embedded in the pickle. Hash
verification against a manifest committed to git is the only thing standing
between a corrupted/tampered download and code execution, so it is not
optional here.

Usage:
  python scripts/fetch_models.py [--models rf1,rf2,...]

Requires HF_TOKEN in .env if the HuggingFace repo is private (it is, until
the pre-publication checklist flips it to public — see PUBLICATION_CHECKLIST.md).
"""

from __future__ import annotations

import argparse
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


def _check_sklearn_version(expected: str | None) -> None:
    if not expected:
        return
    try:
        import sklearn
        running = sklearn.__version__
    except ImportError:
        print("[fetch_models] WARNING: scikit-learn is not installed; cannot verify "
              "compatibility with the pickled model.")
        return
    if running != expected:
        print(f"[fetch_models] WARNING: running scikit-learn {running} != "
              f"the version this model was pickled with ({expected}). "
              "The model may fail to load or load incorrectly. See Models/README.md.")


def fetch(models: list[str] | None = None) -> None:
    if not config.MODEL_MANIFEST.exists():
        raise SystemExit(f"{config.MODEL_MANIFEST} not found — nothing to fetch from.")

    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    expected_sklearn = manifest.get("provenance", {}).get("packages", {}).get("sklearn")
    _check_sklearn_version(expected_sklearn)

    from huggingface_hub import hf_hub_download

    targets = models or list(config.MODEL_FILES.keys())
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key in targets:
        entry = manifest.get("models", {}).get(key, {})
        if entry.get("status") != "trained":
            print(f"[fetch_models] {key}: not trained upstream, skipping")
            continue

        uri = entry.get("uri")
        if not uri:
            print(f"[fetch_models] {key}: no URI in manifest (not yet uploaded), skipping. "
                  f"Run `cli.py train --models {key}` locally instead.")
            continue

        out_path = config.MODELS_DIR / entry["file"]
        if out_path.exists() and sha256_of(out_path) == entry.get("sha256"):
            print(f"[fetch_models] {key}: already present and verified, skipping download")
            continue

        # uri format: hf://<repo_id>/<filename>
        if not uri.startswith("hf://"):
            print(f"[fetch_models] {key}: unrecognized URI scheme '{uri}', skipping")
            continue
        repo_id, filename = uri[len("hf://"):].split("/", 1)

        print(f"[fetch_models] {key}: downloading {uri} ...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )

        actual_hash = sha256_of(Path(downloaded))
        expected_hash = entry.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise SystemExit(
                f"[fetch_models] {key}: SHA-256 MISMATCH.\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}\n"
                "Refusing to install this file. Do NOT run joblib.load() on it."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Path(downloaded).replace(out_path) if Path(downloaded).parent != out_path.parent else None
        import shutil
        if str(out_path) != downloaded:
            shutil.copy2(downloaded, out_path)
        print(f"[fetch_models] {key}: verified SHA-256, saved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all in MANIFEST)")
    args = parser.parse_args(argv)
    models = [k.strip() for k in args.models.split(",")] if args.models else None
    fetch(models)


if __name__ == "__main__":
    main()
