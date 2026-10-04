"""
server.py — FastMCP stdio server for lattice-parameter-prediction.

Exposes ONLY:
  predict_lattice_parameter(composition, reference, models)
  artifact_status()

Build/train/evaluate are NOT exposed here — they can run for minutes to hours,
which would time out a synchronous MCP tool call. Use the CLI (cli.py) or the
lattice-build / lattice-train / lattice-evaluate skills for those instead.
Run from any cwd; sys.path fix ensures engine imports resolve correctly.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# --- sys.path fix -----------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# ---------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP

# Load .env before any engine imports
from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

mcp = FastMCP("lattice-parameter-prediction")


@mcp.tool()
def predict_lattice_parameter(
    composition: str,
    reference: str | None = None,
    models: list[str] | None = None,
) -> dict:
    """
    Predict lattice parameters for a solid-solution composition.

    Parameters
    ----------
    composition : Composition string, e.g. "UN0.5C0.5" or "U1 N0.5 C0.5"
    reference   : Optional mp-id or formula to force the reference structure
                  (skips auto-resolution)
    models      : Optional list of model keys to use (e.g. ["rf1","gbr1"]).
                  Default: all available trained models except Linear Regression,
                  which this tool does not report.

    Returns
    -------
    dict with keys:
      status     : "ok" | "needs_build" | "needs_reference" | "invalid_composition"
                   | "error"
      headline   : {model, values: {a[,b,c]}, units: "Å"}
      table      : [{model_key, model_name, a[,b,c]}]
      reference  : {mp_id, crystal_system, spacegroup_num, basis, source, ...}
      warnings   : [str]

    If status == "needs_build":
      Returns missing artifacts, fetch_hint and build_eta — the agent should ask
      the user yes/no before fetching the deposited binaries with
      scripts/fetch_models.py, or, if they prefer a local refit, running
      /lattice-build + /lattice-train.

    If status == "needs_reference":
      Returns candidates list — the agent should ask which end-member to use,
      then re-call with reference=<chosen>.

    If status == "invalid_composition" or "error":
      Returns a reason (unparseable formula, unknown model key).
    """
    # The prediction runs in a child process with stdin closed. Run in-process, it
    # deadlocks on Windows: while the stdio transport holds a pending read on stdin,
    # the lazy compiled-module imports, matminer's multiprocessing pool and the loky
    # workers that MultiOutputRegressor(n_jobs=-1) starts never return.
    cmd = [sys.executable, str(_HERE / "cli.py"), "predict",
           "--composition", composition, "--json"]
    if reference:
        cmd += ["--reference", reference]
    if models:
        cmd += ["--models", ",".join(models)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", cwd=str(_HERE),
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = proc.stdout
    start = out.find("{")
    if start < 0:
        return {"status": "error",
                "reason": f"prediction process exited {proc.returncode}: "
                          f"{proc.stderr.strip()[-2000:]}"}
    return json.loads(out[start:])


@mcp.tool()
def artifact_status() -> dict:
    """
    Return the current state of all pipeline artifacts and per-stage prerequisites.

    Returns dict with keys:
      build, train, predict  : {ok: bool, missing: [str]}
      artifacts              : {dataset_original, dataset_featurized, ..., models: {rf1:bool,...}}
      stages_completed       : [str]
      build_eta, train_eta   : str
      available_models       : [str]
    """
    from engine import artifacts, config

    build_prereqs   = artifacts.check_prereqs("build")
    train_prereqs   = artifacts.check_prereqs("train")
    predict_prereqs = artifacts.check_prereqs("predict")

    manifest = artifacts.read_manifest()
    stages   = manifest.get("stages", {})

    return {
        "build": {
            "ok":      build_prereqs["ok"],
            "missing": build_prereqs["missing"],
        },
        "train": {
            "ok":      train_prereqs["ok"],
            "missing": train_prereqs["missing"],
        },
        "predict": {
            "ok":               predict_prereqs["ok"],
            "missing":          predict_prereqs["missing"],
            "warnings":         predict_prereqs["warnings"],
            "available_models": predict_prereqs["available_models"],
        },
        "artifacts": {
            "dataset_original":   config.dataset_original().exists(),
            "dataset_featurized": config.DATASET_FEATURIZED.exists(),
            "dataset_training":   config.DATASET_TRAINING.exists(),
            "feature_labels":     config.FEATURELABELS.exists(),
            "ml_feature_labels":  config.ML_FEATURELABELS.exists(),
            "models":             {k: config.model_exists(k) for k in config.MODEL_FILES},
        },
        "stages_completed": list(stages.keys()),
        "build_eta":        predict_prereqs["build_eta"],
        "train_eta":        predict_prereqs["train_eta"],
        "available_models": predict_prereqs["available_models"],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
