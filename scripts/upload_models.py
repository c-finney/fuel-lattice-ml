"""
upload_models.py: upload trained model binaries to a Zenodo deposition and record
the resulting zenodo:// URI, size and SHA-256 in Models/MANIFEST.json.

The binaries are uploaded UNCOMPRESSED. Models/README.md explains why: loading
with mmap_mode="r" suppresses roughly 1,800 joblib and NumPy deprecation warnings
and is about 3.3x faster, and joblib silently ignores mmap_mode on a compressed
file, so compressing here would quietly undo both.

Usage:
  python scripts/upload_models.py --deposition <id> [--models rf1,rf2,...]
  python scripts/upload_models.py --deposition <id> --record-only

Requires ZENODO_TOKEN in .env, with the deposit:write scope. Create the draft
deposition in the Zenodo web interface first and pass its numeric id; this script
deliberately does not create or publish depositions, because publishing mints a
DOI and is not something a script should do without a person present.

--record-only skips the transfer and just rewrites the manifest, which is what to
use if a file was uploaded through the browser.

The five binaries total roughly 14 GB, so a full run is bounded by upload
bandwidth rather than by anything this script does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine import config  # noqa: E402

_CHUNK = 1024 * 1024 * 8
_API = "https://zenodo.org/api"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _token() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set. Put it in .env, or export it. It needs the "
            "deposit:write scope, and it is a credential: do not commit it."
        )
    return token


def _bucket_url(deposition: str, token: str) -> str:
    import urllib.request
    req = urllib.request.Request(
        f"{_API}/deposit/depositions/{deposition}?access_token={token}"
    )
    with urllib.request.urlopen(req) as response:
        meta = json.loads(response.read())
    bucket = meta.get("links", {}).get("bucket")
    if not bucket:
        raise SystemExit(
            f"Deposition {deposition} exposes no bucket link, which usually means it is "
            "already published. Create a new version instead of editing a published one."
        )
    return bucket


def _put_file(bucket: str, path: Path, token: str) -> None:
    import urllib.request
    size = path.stat().st_size
    print(f"[upload_models]   uploading {size / 1e9:.2f} GB")
    with open(path, "rb") as f:
        req = urllib.request.Request(
            f"{bucket}/{path.name}?access_token={token}",
            data=f,
            method="PUT",
        )
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(size))
        with urllib.request.urlopen(req) as response:
            if response.status not in (200, 201):
                raise SystemExit(
                    f"Upload of {path.name} returned HTTP {response.status}."
                )


def upload(deposition: str, models: list[str] | None = None,
           record_only: bool = False) -> None:
    if not config.MODEL_MANIFEST.exists():
        raise SystemExit(f"{config.MODEL_MANIFEST} not found.")

    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    targets = models or list(config.MODEL_FILES.keys())

    token = None if record_only else _token()
    bucket = None if record_only else _bucket_url(deposition, token)

    changed = False
    for key in targets:
        entry = manifest.setdefault("models", {}).setdefault(key, {})
        filename = entry.get("file") or f"{config.MODEL_FILES[key][0]}"
        path = config.MODELS_DIR / filename

        if not path.exists():
            print(f"[upload_models] {key}: {path} is absent, skipping")
            continue

        digest = sha256_of(path)
        size = path.stat().st_size
        print(f"[upload_models] {key}: {filename}, {size} bytes, sha256 {digest[:16]}...")

        if not record_only:
            _put_file(bucket, path, token)

        entry.update({
            "file":       filename,
            "compressed": False,
            "bytes":      size,
            "sha256":     digest,
            "uri":        f"zenodo://{deposition}/{filename}",
            "status":     "trained",
        })
        changed = True

    if changed:
        config.MODEL_MANIFEST.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[upload_models] Manifest rewritten: {config.MODEL_MANIFEST}")
        print("[upload_models] The recorded sizes and hashes are of the LOCAL files. "
              "Re-run scripts/fetch_models.py against the published record before "
              "trusting them, since that is the check that actually exercises the "
              "download path.")
    else:
        print("[upload_models] Nothing changed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deposition", required=True,
                        help="Numeric id of an existing Zenodo draft deposition")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all)")
    parser.add_argument("--record-only", action="store_true",
                        help="Rewrite the manifest without transferring anything")
    args = parser.parse_args(argv)
    models = [k.strip() for k in args.models.split(",")] if args.models else None
    upload(args.deposition, models, args.record_only)


if __name__ == "__main__":
    main()
