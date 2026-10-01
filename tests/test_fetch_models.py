"""
test_fetch_models.py — Guards the archived-binary path in scripts/fetch_models.py.

rf2 is deposited on Zenodo as `IndependentRFModel.joblib.zip`, not as the bare
pickle. The manifest entry names the zip in its URI and carries an `archive`
object with the zip's own size and digest, while `sha256` stays the digest of the
extracted pickle. A URI naming a file the deposit does not hold returns 404 only
after publication, when the file set can no longer change, so the agreement
between the manifest and the archive handling is checked here instead.

Also covered: transient HTTP errors are retried while a 404 is not, a missing digest
or archive member is refused cleanly, and neither manifest writer can drop or contradict
the archive entry.

No network access: `_download` is replaced by a local copy, or `urlopen` by a stub.
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts import fetch_models

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO_ROOT / "Models" / "MANIFEST.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def deposit(tmp_path, monkeypatch):
    """A one-model manifest whose binary is served inside a zip from tmp_path."""
    payload = tmp_path / "src" / "Fake.joblib"
    payload.parent.mkdir()
    payload.write_bytes(b"not a real pickle, only bytes to hash" * 1000)
    archive = tmp_path / "src" / "Fake.joblib.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(payload, "Fake.joblib")

    entry = {
        "file": "Fake.joblib",
        "compressed": False,
        "bytes": payload.stat().st_size,
        "sha256": _sha(payload),
        "uri": "https://example.invalid/Fake.joblib.zip",
        "archive": {"format": "zip", "member": "Fake.joblib",
                    "bytes": archive.stat().st_size, "sha256": _sha(archive)},
        "status": "trained",
    }
    manifest = tmp_path / "MANIFEST.json"
    models_dir = tmp_path / "binaries"

    def write(e):
        manifest.write_text(json.dumps({"models": {"fake": e}}), encoding="utf-8")

    write(entry)
    monkeypatch.setattr(fetch_models.config, "MODEL_MANIFEST", manifest)
    monkeypatch.setattr(fetch_models.config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(fetch_models, "_download",
                        lambda url, dest: shutil.copyfile(archive, dest))
    return entry, write, models_dir


def test_archived_binary_is_verified_extracted_and_the_zip_removed(deposit):
    entry, _, models_dir = deposit
    fetch_models.fetch(["fake"])
    out = models_dir / "Fake.joblib"
    assert _sha(out) == entry["sha256"]
    assert sorted(p.name for p in models_dir.iterdir()) == ["Fake.joblib"]


def test_archive_digest_mismatch_stops_before_extraction(deposit):
    entry, write, models_dir = deposit
    write({**entry, "archive": {**entry["archive"], "sha256": "0" * 64}})
    with pytest.raises(SystemExit, match="MISMATCH on the archive"):
        fetch_models.fetch(["fake"])
    assert list(models_dir.iterdir()) == []


def test_extracted_digest_mismatch_deletes_the_extracted_file(deposit):
    entry, write, models_dir = deposit
    write({**entry, "sha256": "0" * 64})
    with pytest.raises(SystemExit, match="MISMATCH"):
        fetch_models.fetch(["fake"])
    assert list(models_dir.iterdir()) == []


def test_missing_archive_member_is_a_clean_exit(deposit):
    entry, write, models_dir = deposit
    write({**entry, "archive": {**entry["archive"], "member": "Other.joblib"}})
    with pytest.raises(SystemExit, match="no member named 'Other.joblib'"):
        fetch_models.fetch(["fake"])
    assert list(models_dir.iterdir()) == []


def test_entry_without_digest_is_refused(deposit):
    entry, write, models_dir = deposit
    write({k: v for k, v in entry.items() if k != "sha256"})
    with pytest.raises(SystemExit, match="no SHA-256"):
        fetch_models.fetch(["fake"])
    assert list(models_dir.iterdir()) == []


class _Response:
    """Just enough of an HTTP response for _download()."""

    def __init__(self, body: bytes):
        self.status, self._body = 200, body
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n):
        out, self._body = self._body[:n], self._body[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen_failing_with(codes, body, calls):
    def urlopen(req, timeout=None):
        calls.append(timeout)
        if len(calls) <= len(codes):
            raise fetch_models.urllib.error.HTTPError(req.full_url, codes[len(calls) - 1],
                                                      "error", {}, None)
        return _Response(body)
    return urlopen


def test_transient_http_errors_are_retried(tmp_path, monkeypatch):
    calls, body = [], b"x" * 1000
    monkeypatch.setattr(fetch_models, "_BACKOFF", 0)
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen",
                        _urlopen_failing_with([503, 429], body, calls))
    dest = tmp_path / "f.bin"
    fetch_models._download("https://example.invalid/f.bin", dest)
    assert dest.read_bytes() == body
    assert len(calls) == 3 and all(t == fetch_models._TIMEOUT for t in calls)
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_not_found_is_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch_models, "_BACKOFF", 0)
    monkeypatch.setattr(fetch_models.urllib.request, "urlopen",
                        _urlopen_failing_with([404], b"", calls))
    with pytest.raises(fetch_models.urllib.error.HTTPError):
        fetch_models._download("https://example.invalid/f.bin", tmp_path / "f.bin")
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_record_only_upload_keeps_the_archive_uri(deposit, monkeypatch):
    from scripts import upload_models

    entry, _, models_dir = deposit
    models_dir.mkdir()
    (models_dir / "Fake.joblib").write_bytes(b"not a real pickle, only bytes to hash" * 1000)
    upload_models.upload("123", ["fake"], record_only=True)
    after = json.loads(fetch_models.config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    rec = after["models"]["fake"]
    assert rec["uri"] == "zenodo://123/Fake.joblib.zip"
    assert rec["archive"] == entry["archive"]
    assert rec["sha256"] == entry["sha256"]


def test_regenerated_manifest_keeps_the_archive(deposit, monkeypatch):
    from scripts import write_model_manifest

    entry, _, _ = deposit
    fresh = {"file": entry["file"], "compressed": False, "bytes": entry["bytes"],
             "sha256": entry["sha256"], "uri": None, "status": "trained"}
    monkeypatch.setattr(write_model_manifest, "build_manifest",
                        lambda: {"provenance": {}, "models": {"fake": dict(fresh)}})
    write_model_manifest.main()
    after = json.loads(fetch_models.config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    rec = after["models"]["fake"]
    assert rec["uri"] == entry["uri"]
    assert rec["archive"] == entry["archive"]


def test_manifest_uris_name_what_is_deposited():
    """Each URI names the bare binary, or for an archived entry the zip of it."""
    for key, entry in MANIFEST["models"].items():
        if entry.get("status") != "trained":
            continue
        name = entry["uri"].rsplit("/", 1)[1]
        if "archive" in entry:
            assert entry["archive"]["format"] == "zip", key
            assert entry["archive"]["member"] == entry["file"], key
            assert name == entry["file"] + ".zip", key
            assert len(entry["archive"]["sha256"]) == 64, key
        else:
            assert name == entry["file"], key


def test_rf2_is_the_archived_entry():
    assert "archive" in MANIFEST["models"]["rf2"]
    assert [k for k, e in MANIFEST["models"].items() if "archive" in e] == ["rf2"]
