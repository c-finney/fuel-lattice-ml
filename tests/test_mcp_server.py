"""
test_mcp_server.py — Guards the agent-tool path, which the manuscript's Software
and Code Availability statement advertises by name and which no other test
exercises. Two shipped defects lived in that gap:

1. `requirements.txt` asked for `mcp>=1.2` with no upper bound. A clean install
   resolved to 2.x, where FastMCP was renamed to MCPServer and
   `mcp.server.fastmcp` stopped existing, so `server.py` failed at import on any
   fresh clone. Nothing imported `server.py`, so the suite passed anyway.

2. `.mcp.json` declared `env = {"LATTICE_DATA_ROOT": "${LATTICE_DATA_ROOT}"}`.
   A launcher substitutes nothing for an unset variable, so the server received
   the literal string. `Path("${LATTICE_DATA_ROOT}")` is a relative directory of
   that name, so DATA_ROOT moved off REPO_ROOT, every binary was reported
   missing, and importing engine.config created the directory tree under it.
   A reader was told to spend about seventy minutes rebuilding artifacts that
   were already present.

These tests must stay correct in a clean clone with no model binaries present,
so nothing here asserts that any binary exists. What is asserted is that the
server's report agrees with what is actually on disk, whatever that is.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# A deliberately unexpanded value, spelled at runtime so this file does not
# itself contain the literal a launcher would emit.
UNEXPANDED = "${" + "LATTICE_DATA_ROOT}"


# ---------------------------------------------------------------------------
# 1. The server imports and exposes the tools it claims
# ---------------------------------------------------------------------------

def test_server_module_imports():
    """Catches defect 1: an mcp major-version bump moving FastMCP.

    Imported directly rather than behind a skipif, because skipping when `mcp`
    is absent or incompatible would hide the exact breakage this guards.
    """
    import server

    assert server.mcp is not None


def test_server_exposes_exactly_its_two_documented_tools():
    import asyncio

    import server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {"predict_lattice_parameter", "artifact_status"}, names


# ---------------------------------------------------------------------------
# 2. DATA_ROOT ignores values a launcher never filled in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["", "   ", UNEXPANDED, "  " + UNEXPANDED + "  ", "${MP_API_KEY}"],
    ids=["empty", "whitespace", "unexpanded", "unexpanded-padded", "other-unexpanded"],
)
def test_env_override_treats_unusable_values_as_absent(value, monkeypatch):
    """Catches defect 2 at the unit level."""
    from engine import config

    monkeypatch.setenv("LATTICE_DATA_ROOT", value)
    assert config._env_override("LATTICE_DATA_ROOT") is None


def test_env_override_is_absent_when_unset(monkeypatch):
    from engine import config

    monkeypatch.delenv("LATTICE_DATA_ROOT", raising=False)
    assert config._env_override("LATTICE_DATA_ROOT") is None


def test_env_override_honours_a_real_value(monkeypatch, tmp_path):
    """The override is a supported feature and must keep working."""
    from engine import config

    monkeypatch.setenv("LATTICE_DATA_ROOT", str(tmp_path))
    assert config._env_override("LATTICE_DATA_ROOT") == str(tmp_path)


def _data_root_in_subprocess(env_value, cwd):
    """Import engine.config fresh under a given env and report its DATA_ROOT.

    A subprocess is required because DATA_ROOT is resolved once, at import time,
    and this suite has already imported the module.
    """
    env = dict(os.environ)
    if env_value is None:
        env.pop("LATTICE_DATA_ROOT", None)
    else:
        env["LATTICE_DATA_ROOT"] = env_value
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "from engine import config; print(config.DATA_ROOT)" % REPO_ROOT],
        capture_output=True, text=True, env=env, cwd=str(cwd), check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize(
    "env_value", [None, "", UNEXPANDED], ids=["unset", "empty", "unexpanded"]
)
def test_data_root_falls_back_to_repo_root(env_value, tmp_path):
    assert _data_root_in_subprocess(env_value, tmp_path) == str(REPO_ROOT)


def test_unexpanded_value_creates_no_stray_directory(tmp_path):
    """The litter half of defect 2.

    engine.config mkdirs its regenerable artifact dirs at import time, so a
    DATA_ROOT of "${LATTICE_DATA_ROOT}" created that tree under the process's
    working directory. Run from an empty directory, nothing may appear in it.
    """
    _data_root_in_subprocess(UNEXPANDED, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_real_override_still_moves_the_artifact_root(tmp_path):
    """Guards against fixing the above by ignoring the variable altogether."""
    alt = tmp_path / "alt-root"
    alt.mkdir()
    assert _data_root_in_subprocess(str(alt), tmp_path) == str(alt)


# ---------------------------------------------------------------------------
# 3. The status report agrees with the filesystem
# ---------------------------------------------------------------------------

def test_artifact_status_has_its_documented_shape():
    import server

    status = server.artifact_status()
    for key in ("build", "train", "predict", "artifacts", "available_models"):
        assert key in status, key
    for stage in ("build", "train", "predict"):
        assert set(status[stage]) >= {"ok", "missing"}
        assert isinstance(status[stage]["ok"], bool)


def test_artifact_status_model_flags_match_what_is_on_disk():
    """The regression that defect 2 actually produced.

    Vacuous in a clean clone with no binaries, and a real check whenever any
    binary is present: a model may be reported available only if its file is
    genuinely there, and must be reported available if it is.
    """
    import server
    from engine import config

    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    reported = server.artifact_status()["artifacts"]["models"]

    for key, entry in manifest["models"].items():
        on_disk = (config.MODELS_DIR / entry["file"]).exists()
        assert reported[key] == on_disk, (
            f"{key}: status says {reported[key]}, "
            f"{config.MODELS_DIR / entry['file']} exists={on_disk}"
        )


def test_available_models_are_a_subset_of_those_present():
    import server
    from engine import config

    manifest = json.loads(config.MODEL_MANIFEST.read_text(encoding="utf-8"))
    status = server.artifact_status()
    present = {
        k for k, e in manifest["models"].items()
        if (config.MODELS_DIR / e["file"]).exists()
    }
    assert set(status["available_models"]) <= present


# ---------------------------------------------------------------------------
# 4. The launcher config that caused defect 2
# ---------------------------------------------------------------------------

def test_mcp_json_declares_no_unexpanded_placeholders():
    """`.mcp.json` must not hand the server a variable reference it cannot fill.

    engine.config loads .env at import and a real process variable is inherited
    by the child anyway, so declaring these is unnecessary as well as unsafe:
    MP_API_KEY had the same defect, where the literal would have displaced the
    real key, because dotenv does not override a variable already set.
    """
    raw = (REPO_ROOT / ".mcp.json").read_text(encoding="utf-8")
    assert "${" not in raw, "unexpanded ${...} placeholder in .mcp.json"

    server_cfg = json.loads(raw)["mcpServers"]["lattice-prediction"]
    for name, value in (server_cfg.get("env") or {}).items():
        assert not value.strip().startswith("${"), f"{name} is an unexpanded reference"
