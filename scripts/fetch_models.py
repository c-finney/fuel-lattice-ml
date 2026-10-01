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

An entry may also carry an "archive" object, {"format": "zip", "member", "bytes",
"sha256"}, when the URI names a zip holding the binary rather than the binary
itself. rf2 is deposited this way. The archive's digest is checked first, the
named member is then extracted, and the extracted file is checked against the
entry's "sha256" exactly as an unarchived download is. That second check is the
one that gates the unpickle. The member is a plain uncompressed pickle, so
mmap_mode works on it as on the others.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
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
    # requirements.txt pins scikit-learn==1.9.*, and RELEASE.md records the
    # deposited models as verified across that patch range, so only a change of
    # major or minor version is worth a warning.
    if running.split(".")[:2] != expected.split(".")[:2]:
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


_MAX_ATTEMPTS = 6   # consecutive attempts without progress before giving up
_MAX_TOTAL = 60     # attempts in all, however much each one progressed
_BACKOFF = 5  # seconds, multiplied by the consecutive-failure count
_TIMEOUT = 60  # seconds without data before a stalled connection is retried
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _download(url: str, dest: Path) -> None:
    """
    Stream *url* to *dest*, resuming on a dropped connection.

    A multi-gigabyte transfer over one TLS connection does not reliably survive.
    The failure seen in practice is an SSL EOF part way through, which arrives as
    URLError rather than as a short read, so a plain single-shot download of rf1
    (3.97 GB) or the rf2 archive (2.60 GB) can fail repeatedly on an otherwise
    healthy link.

    Each attempt therefore resumes from the bytes already on disk using a Range
    request, so a drop at 3 GB costs the remainder rather than the whole file. A
    server that ignores Range answers 200 instead of 206, which is handled by
    restarting the file rather than appending to it and corrupting it silently.
    Integrity is not assumed from any of this: fetch() verifies SHA-256 against
    the manifest afterwards, and that check is what actually gates the unpickle.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp hands back an OPEN os-level descriptor. Closing it immediately is
    # required, not tidiness: Windows refuses to unlink a file that still has an
    # open handle, so leaving it open makes the cleanup below raise
    # PermissionError (WinError 32) and bury whatever actually went wrong --
    # a 404 from an unpublished deposit, most likely -- under a traceback about
    # a temp file. POSIX allows unlinking an open file, so this never surfaces
    # on Linux.
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    os.close(fd)
    tmp = Path(tmp_name)
    total = None
    # Consecutive attempts that made no progress. An attempt that added bytes
    # resets the count, so a long download over a link that drops repeatedly
    # still finishes; _MAX_TOTAL bounds the whole loop regardless.
    failures = 0
    try:
        for attempt in range(1, _MAX_TOTAL + 1):
            have = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as response:
                    # 206 means the server honoured the Range and we append;
                    # 200 means it sent the whole file, so start over.
                    resuming = response.status == 206
                    if have and not resuming:
                        have = 0
                    if total is None:
                        length = response.headers.get("Content-Length")
                        crange = response.headers.get("Content-Range")
                        if crange and "/" in crange:
                            total = int(crange.rsplit("/", 1)[1])
                        elif length:
                            total = int(length) + have
                    seen = have
                    with open(tmp, "ab" if resuming and have else "wb") as out:
                        while True:
                            chunk = response.read(_CHUNK)
                            if not chunk:
                                break
                            out.write(chunk)
                            seen += len(chunk)
                            if total:
                                print(f"\r[fetch_models]   {seen / 1e9:6.2f} / "
                                      f"{total / 1e9:.2f} GB", end="", flush=True)
                if total:
                    print()
                if total is None or tmp.stat().st_size >= total:
                    break
                raise urllib.error.URLError(
                    f"short read: {tmp.stat().st_size} of {total} bytes")
            except urllib.error.HTTPError as exc:
                # A 404 will not change on retry. Rate limiting and gateway
                # errors are transient, so they keep the partial file and
                # resume like a dropped connection.
                failures += 1
                if exc.code not in _RETRYABLE_HTTP or failures >= _MAX_ATTEMPTS:
                    raise
                got = tmp.stat().st_size if tmp.exists() else 0
                print(f"\n[fetch_models]   attempt {attempt} got HTTP {exc.code}; "
                      f"resuming from {got / 1e9:.2f} GB in {_BACKOFF * failures}s")
                time.sleep(_BACKOFF * failures)
            except (urllib.error.URLError, OSError, EOFError,
                    http.client.IncompleteRead) as exc:
                got = tmp.stat().st_size if tmp.exists() else 0
                failures = 1 if got > have else failures + 1
                if failures >= _MAX_ATTEMPTS or attempt == _MAX_TOTAL:
                    raise
                print(f"\n[fetch_models]   attempt {attempt} failed ({exc}); "
                      f"resuming from {got / 1e9:.2f} GB in {_BACKOFF * failures}s")
                time.sleep(_BACKOFF * failures)
        # os.replace renames over an existing file on every platform, where
        # shutil.move falls back to copying the whole file on Windows.
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verify(key: str, path: Path, expected: str | None, what: str = "") -> None:
    """Delete *path* and stop if its SHA-256 is not *expected*.

    A missing digest is refused rather than passed, since this check is what
    stands between the download and joblib.load().
    """
    if not expected:
        path.unlink(missing_ok=True)
        raise SystemExit(f"[fetch_models] {key}: the manifest records no SHA-256{what}, "
                         "so the download cannot be verified and has been deleted.")
    actual = sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"[fetch_models] {key}: SHA-256 MISMATCH{what}.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "The downloaded file has been deleted. Do NOT run joblib.load() on a "
            "copy of it."
        )


def _extract(archive_path: Path, archive: dict, dest: Path) -> None:
    """Extract the one named member of a zip archive to *dest*.

    Only the member the manifest names is read, and it is written to *dest*
    rather than to whatever path the archive records, so an archive cannot
    place a file anywhere else.
    """
    fmt = archive.get("format")
    if fmt != "zip":
        raise SystemExit(f"[fetch_models] unsupported archive format '{fmt}'")
    member = archive.get("member") or dest.name
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".extract-")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            if member not in zf.namelist():
                raise SystemExit(f"[fetch_models] the archive holds no member named "
                                 f"'{member}'; it holds {zf.namelist()}")
            with zf.open(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _in_use(key: str, exc: OSError) -> str:
    return (f"[fetch_models] {key}: could not write {exc.filename or 'the model file'} "
            f"({exc.strerror or exc}). On Windows this usually means the existing file is "
            "open in another program, such as a running MCP server or notebook that has "
            "loaded the model. Close it and run the command again.")


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

        archive = entry.get("archive")
        # Refuse before downloading anything, and before touching a file that may
        # already be in place, if there is nothing to verify the download against.
        if not expected_hash or (archive and not archive.get("sha256")):
            raise SystemExit(f"[fetch_models] {key}: the manifest records no SHA-256"
                             f"{' for the archive' if expected_hash else ''}, so a download "
                             "could not be verified. Nothing was downloaded.")
        download_path = out_path.with_name(out_path.name + ".zip") if archive else out_path
        size = (archive or entry).get("bytes")
        size_note = f" ({size / 1e9:.2f} GB)" if size else ""
        print(f"[fetch_models] {key}: downloading{size_note} from {url}")
        try:
            _download(url, download_path)
        except urllib.error.HTTPError as exc:
            hint = ("The file was not found at that address." if exc.code == 404 else
                    "Try again later." if exc.code in _RETRYABLE_HTTP else
                    "The server refused the request.")
            raise SystemExit(
                f"[fetch_models] {key}: download failed with HTTP {exc.code}. {hint}"
            ) from exc
        except PermissionError as exc:
            raise SystemExit(_in_use(key, exc)) from exc
        except (urllib.error.URLError, OSError, http.client.IncompleteRead) as exc:
            raise SystemExit(
                f"[fetch_models] {key}: download failed after repeated attempts ({exc}). "
                "Check the connection and run the command again."
            ) from exc

        if archive:
            _verify(key, download_path, archive.get("sha256"), " on the archive")
            print(f"[fetch_models] {key}: archive verified, extracting {entry['file']}")
            try:
                _extract(download_path, archive, out_path)
            except PermissionError as exc:
                raise SystemExit(_in_use(key, exc)) from exc
            finally:
                download_path.unlink(missing_ok=True)

        _verify(key, out_path, expected_hash)
        print(f"[fetch_models] {key}: verified SHA-256, saved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all in MANIFEST)")
    args = parser.parse_args(argv)
    models = [k.strip() for k in args.models.split(",")] if args.models else None
    fetch(models)


if __name__ == "__main__":
    main()
