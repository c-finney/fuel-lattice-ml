"""
cli.py — Single entry point for all lattice-parameter-prediction slash commands.

Usage:
  python cli.py status [--json]
  python cli.py build  [--thresh N] [--resume] [--force] [--limit N]
  python cli.py train  [--fast | --full | --models a,b,...] [--n-jobs N]
  python cli.py evaluate [--models ...] [--out-dir DIR]
  python cli.py predict --composition COMP | --csv PATH
                        [--reference REF] [--models a,b,...] [--out-dir DIR] [--json]

cli.py prepends its own directory to sys.path so it works from any cwd.
"""

import sys
import os
from pathlib import Path

# --- sys.path fix: always resolve engine imports relative to THIS file ------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# ---------------------------------------------------------------------------

import argparse
import json


def _cmd_status(rest: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="cli.py status")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args(rest)

    from engine import artifacts, config

    build_prereqs   = artifacts.check_prereqs("build")
    train_prereqs   = artifacts.check_prereqs("train")
    predict_prereqs = artifacts.check_prereqs("predict")

    manifest = artifacts.read_manifest()
    stages   = manifest.get("stages", {})

    status = {
        "build": {
            "ok":      build_prereqs["ok"],
            "missing": build_prereqs["missing"],
        },
        "train": {
            "ok":      train_prereqs["ok"],
            "missing": train_prereqs["missing"],
        },
        "predict": {
            "ok":              predict_prereqs["ok"],
            "missing":         predict_prereqs["missing"],
            "warnings":        predict_prereqs["warnings"],
            "available_models": predict_prereqs["available_models"],
        },
        "artifacts": {
            "dataset_original":  config.dataset_original().exists(),
            "dataset_featurized": config.DATASET_FEATURIZED.exists(),
            "dataset_training":  config.DATASET_TRAINING.exists(),
            "feature_labels":    config.FEATURELABELS.exists(),
            "ml_feature_labels": config.ML_FEATURELABELS.exists(),
            "models":            {k: config.model_exists(k) for k in config.MODEL_FILES},
        },
        "stages_completed": list(stages.keys()),
        "build_eta":  predict_prereqs["build_eta"],
        "train_eta":  predict_prereqs["train_eta"],
    }

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_status(status)


def _print_status(s: dict) -> None:
    ok   = lambda b: "OK" if b else "MISSING"
    tick = lambda b: "[x]" if b else "[ ]"

    print("=" * 60)
    print("  Lattice Parameter Prediction — Artifact Status")
    print("=" * 60)

    print("\n[Prereqs]")
    print(f"  build   : {ok(s['build']['ok'])}")
    for m in s["build"].get("missing", []):
        print(f"           ! {m}")
    print(f"  train   : {ok(s['train']['ok'])}")
    for m in s["train"].get("missing", []):
        print(f"           ! {m}")
    print(f"  predict : {ok(s['predict']['ok'])}")
    for m in s["predict"].get("missing", []):
        print(f"           ! {m}")
    for w in s["predict"].get("warnings", []):
        print(f"           (i) {w}")

    print("\n[Artifacts]")
    art = s["artifacts"]
    print(f"  {tick(art['dataset_original'])}  MP_Dataset_Original_Trimmed.csv")
    print(f"  {tick(art['dataset_featurized'])} MP_Dataset_Featurized.csv")
    print(f"  {tick(art['dataset_training'])}  Training_Dataset.csv")
    print(f"  {tick(art['feature_labels'])}    FeatureLabels.joblib")
    print(f"  {tick(art['ml_feature_labels'])} ML_FeatureLabels.joblib")
    print("\n[Models]")
    for k, exists in art["models"].items():
        print(f"  {tick(exists)}  {k}")

    if s["stages_completed"]:
        print(f"\n[Completed stages] {', '.join(s['stages_completed'])}")
    else:
        print("\n[Completed stages] (none — nothing built yet)")

    print(f"\n[Build ETA]  {s['build_eta']}")
    print(f"[Train ETA]  {s['train_eta']}")
    print("=" * 60)


def _cmd_build(rest: list[str]) -> None:
    from engine.build_dataset import main
    main(rest)


def _cmd_train(rest: list[str]) -> None:
    from engine.train_models import main
    main(rest)


def _cmd_evaluate(rest: list[str]) -> None:
    from engine.evaluate_cv import main
    main(rest)


def _cmd_predict(rest: list[str]) -> None:
    from engine.predict import main
    main(rest)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "status":   _cmd_status,
    "build":    _cmd_build,
    "train":    _cmd_train,
    "evaluate": _cmd_evaluate,
    "predict":  _cmd_predict,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: cli.py <{'|'.join(COMMANDS)}> [args...]")
        print("\nCommands:")
        print("  status    — show artifact/prereq state")
        print("  build     — build featurized MP dataset (notebook 1)")
        print("  train     — train models (notebook 2)")
        print("  evaluate  — cross-validation metrics (notebook 2 CV)")
        print("  predict   — predict lattice parameters (notebook 3)")
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]
    COMMANDS[cmd](rest)


if __name__ == "__main__":
    main()
