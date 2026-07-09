"""
upload_models.py — Upload trained model binaries to a private HuggingFace
repo and record the resulting hf:// URI in Models/MANIFEST.json.

The model binary is uploaded UNCOMPRESSED — see Models/README.md for why
(mmap_mode="r" loading, which suppresses ~1,800 joblib/NumPy warnings and is
~3.3x faster, silently stops working on a compressed joblib file).

Usage:
  python scripts/upload_models.py --repo-id <hf-namespace>/<repo-name> [--models rf1,...] [--private]

Requires HF_TOKEN in .env (or an active `hf auth login` session).
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


def upload(repo_id: str, models: list[str] | None = None, private: bool = True) -> None:
    from huggingface_hub import HfApi

    if not config.MODEL_MANIFEST.exists():
        raise SystemExit(
            f"{config.MODEL_MANIFEST} not found — run scripts/write_model_manifest.py first."
        )
    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[upload_models] repo ready: {repo_id} (private={private})")

    targets = models or list(config.MODEL_FILES.keys())
    changed = False

    for key in targets:
        entry = manifest.get("models", {}).get(key, {})
        if entry.get("status") != "trained":
            print(f"[upload_models] {key}: not trained, skipping")
            continue

        path = config.MODELS_DIR / entry["file"]
        if not path.exists():
            print(f"[upload_models] {key}: {path} not found on disk, skipping")
            continue

        # Verify the binary is uncompressed before uploading — the whole
        # point of shipping it that way is the mmap_mode="r" load path.
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic[:2] in (b"\x1f\x8b",) or magic[:3] == b"BZh":
            raise SystemExit(
                f"[upload_models] {key}: {path} appears to be gzip/bzip2 compressed. "
                "The shipped model must stay UNCOMPRESSED — see Models/README.md."
            )

        local_hash = sha256_of(path)
        if entry.get("sha256") and entry["sha256"] != local_hash:
            print(f"[upload_models] {key}: WARNING — on-disk hash does not match "
                  "MANIFEST.json. Re-run scripts/write_model_manifest.py before uploading.")

        print(f"[upload_models] {key}: uploading {path} "
              f"({path.stat().st_size:,} bytes) — this can take a while...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=entry["file"],
            repo_id=repo_id,
            repo_type="model",
        )

        uri = f"hf://{repo_id}/{entry['file']}"
        manifest["models"][key]["uri"] = uri
        manifest["models"][key]["sha256"] = local_hash
        changed = True
        print(f"[upload_models] {key}: done -> {uri}")

    if changed:
        config.MODEL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[upload_models] Updated {config.MODEL_MANIFEST}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo id, e.g. c-finney/fuel-lattice-ml")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all trained)")
    parser.add_argument("--public", action="store_true",
                        help="Create/use a PUBLIC repo instead of private (default: private)")
    args = parser.parse_args(argv)
    models = [k.strip() for k in args.models.split(",")] if args.models else None
    upload(args.repo_id, models=models, private=not args.public)


if __name__ == "__main__":
    main()
