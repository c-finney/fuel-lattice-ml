"""
fetch_models.py: download trained model binaries from the archive recorded in
Models/MANIFEST.json, verifying SHA-256 BEFORE unpickling.

joblib.load() executes arbitrary code embedded in the pickle. Hash verification
against a manifest committed to git is the only thing standing between a
corrupted or tampered download and code execution, so it is not optional here.

The binaries live in the Zenodo deposit that archives this repository, which is
open and needs no account, no token and no login. Nothing in this script
authenticates.

Usage:
  python scripts/fetch_models.py [--models rf1,rf2,...]

The manifest carries one URI per model, in either of two forms:

  zenodo://<record_id>/<filename>
  https://<any direct download URL>

The zenodo:// form is expanded to the record's file endpoint. Both are fetched
over plain HTTPS by urllib, so no third-party client library is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine import config  # noqa: E402

_CHUNK = 1024 * 1024 * 8


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sklearn_version(expected: str | None) -> None:
    if not expected:
        return
    try:
        import sklearn
        running = sklearn.__version__
    except ImportError:
        print("[fetch_models] WARNING: scikit-learn is not installed, so compatibility "
              "with the pickled model cannot be verified.")
        return
    if running != expected:
        print(f"[fetch_models] WARNING: running scikit-learn {running} against a model "
              f"pickled with {expected}. It may fail to load, or load incorrectly. "
              "See Models/README.md.")


def _resolve_uri(uri: str) -> str | None:
    """Turn a manifest URI into an HTTPS URL, or None if the scheme is unknown."""
    if uri.startswith("https://"):
        return uri
    if uri.startswith("zenodo://"):
        rest = uri[len("zenodo://"):]
        record_id, _, filename = rest.partition("/")
        if not record_id or not filename:
            return None
        return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    return None


def _download(url: str, dest: Path) -> None:
    """Stream *url* to *dest*, hashing nothing; verification happens afterwards."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(dir=dest.parent, prefix=".download-")[1])
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            seen = 0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r[fetch_models]   {seen / 1e9:6.2f} / {total / 1e9:.2f} GB",
                          end="", flush=True)
            if total:
                print()
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch(models: list[str] | None = None) -> None:
    if not config.MODEL_MANIFEST.exists():
        raise SystemExit(f"{config.MODEL_MANIFEST} not found, so there is nothing to fetch from.")

    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    _check_sklearn_version(
        manifest.get("provenance", {}).get("packages", {}).get("sklearn")
    )

    targets = models or list(config.MODEL_FILES.keys())
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key in targets:
        entry = manifest.get("models", {}).get(key, {})
        if entry.get("status") != "trained":
            print(f"[fetch_models] {key}: not trained upstream, skipping")
            continue

        uri = entry.get("uri")
        if not uri:
            print(f"[fetch_models] {key}: no URI in the manifest, so it has not been "
                  f"uploaded. Train it locally with `cli.py train --models {key}`.")
            continue

        url = _resolve_uri(uri)
        if url is None:
            print(f"[fetch_models] {key}: unrecognized URI '{uri}', skipping")
            continue

        out_path = config.MODELS_DIR / entry["file"]
        expected_hash = entry.get("sha256")

        if out_path.exists() and expected_hash and sha256_of(out_path) == expected_hash:
            print(f"[fetch_models] {key}: already present and verified, skipping download")
            continue

        size = entry.get("bytes")
        size_note = f" ({size / 1e9:.2f} GB)" if size else ""
        print(f"[fetch_models] {key}: downloading{size_note} from {url}")
        try:
            _download(url, out_path)
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"[fetch_models] {key}: download failed with HTTP {exc.code}. "
                "If the Zenodo deposit is not published yet, the record will 404; "
                "see RELEASE.md."
            ) from exc

        actual_hash = sha256_of(out_path)
        if expected_hash and actual_hash != expected_hash:
            out_path.unlink(missing_ok=True)
            raise SystemExit(
                f"[fetch_models] {key}: SHA-256 MISMATCH.\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}\n"
                "The downloaded file has been deleted. Do NOT run joblib.load() on a "
                "copy of it."
            )
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
