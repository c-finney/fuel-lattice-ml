"""
test_fetch_models.py — Guards the archived-binary path in scripts/fetch_models.py.

rf2 is deposited on Zenodo as `IndependentRFModel.joblib.zip`, not as the bare
pickle. The manifest entry names the zip in its URI and carries an `archive`
object with the zip's own size and digest, while `sha256` stays the digest of the
extracted pickle. A URI naming a file the deposit does not hold returns 404 only
after publication, when the file set can no longer change, so the agreement
between the manifest and the archive handling is checked here instead.

No network access: `_download` is replaced by a local copy.
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
