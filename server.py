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

import sys
import os
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
                  Default: all available trained models except Linear Regression.

    Returns
    -------
    dict with keys:
      status     : "ok" | "needs_build" | "needs_reference"
      headline   : {model, values: {a[,b,c]}, units: "Å"}
      table      : [{model_key, model_name, a[,b,c]}]
      reference  : {mp_id, crystal_system, spacegroup_num, basis, source, ...}
      warnings   : [str]

    If status == "needs_build":
      Returns missing artifacts and build_eta — the agent should ask
      the user yes/no before running /lattice-build + /lattice-train.

    If status == "needs_reference":
      Returns candidates list — the agent should ask which end-member to use,
      then re-call with reference=<chosen>.
    """
    from engine.predict import predict_one
    return predict_one(composition, reference=reference, models=models)


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
