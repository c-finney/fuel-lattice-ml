"""
test_params_parity.py — Guards decision "engine/ is the single source of
truth for hyperparameters" (see engine/train_models.py's model_estimators()).

1. Every Models/<Name>/params.json's declared params must match what
   engine.train_models.model_estimators() actually constructs.
2. No hyperparameter-looking assignment may appear hardcoded in any of the
   four core notebooks at the repository root — they must import engine
   instead. This is exactly the mechanism that let the (unmaintained)
   TrainTest notebook drift from CrossVal on gbr2's max_iter (1500 vs 1800)
   before this repository existed.
3. No notebook may contain a leftover local-machine path or the revoked
   Materials Project API key.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DIR_TO_KEY = {
    "DependentRFModel": "rf1",
    "IndependentRFModel": "rf2",
    "XGBoostGBRModel": "gbr1",
    "ScikitLearnGBRModel": "gbr2",
    "LinearRegressionModel": "lin",
}

CORE_NOTEBOOKS = [
    "FuelLatticeParameterDataFeaturization.ipynb",
    "FuelLatticeParameterModelCreation_CrossVal.ipynb",
    "FuelLatticeParameterModelPrediction.ipynb",
    "FuelLatticeParameterModelOptimization.ipynb",
]

HYPERPARAM_PATTERN = re.compile(
    r"\b(n_estimators|max_iter|learning_rate|max_depth|random_state)\s*="
)
LOCAL_PATH_PATTERN = re.compile(r"C:[/\\]Users")
REVOKED_KEY = "csvp7B7"


def _estimator_params(est) -> dict:
    """
    Return get_params() of the actual fitted estimator, unwrapping
    MultiOutputRegressor to its prototype .estimator where present.

    Must check isinstance(MultiOutputRegressor), NOT hasattr(est, "estimator"):
    modern scikit-learn's own ensemble estimators (e.g. RandomForestRegressor)
    also expose an `.estimator` attribute — the base tree prototype — so a
    hasattr check would wrongly unwrap rf1/gbr1 too and inspect the wrong
    object's params.
    """
    from sklearn.multioutput import MultiOutputRegressor

    if isinstance(est, MultiOutputRegressor):
        return est.estimator.get_params()
    return est.get_params()


class TestModelParamsParity:

    @pytest.mark.parametrize("dirname,key", list(DIR_TO_KEY.items()))
    def test_params_json_matches_model_estimators(self, dirname, key):
        from engine.train_models import model_estimators

        params_path = REPO_ROOT / "Models" / dirname / "params.json"
        assert params_path.exists(), f"missing {params_path}"
        declared = json.loads(params_path.read_text(encoding="utf-8"))

        estimators = model_estimators()
        assert key in estimators, f"model_estimators() has no key '{key}'"

        actual_params = _estimator_params(estimators[key])
        for pname, pvalue in declared.get("params", {}).items():
            assert pname in actual_params, (
                f"{dirname}/params.json declares '{pname}' but "
                f"model_estimators()['{key}'] has no such param"
            )
            assert actual_params[pname] == pvalue, (
                f"{dirname}/params.json says {pname}={pvalue!r}, but "
                f"model_estimators()['{key}'] has {pname}={actual_params[pname]!r}. "
                "engine/train_models.py is the single source of truth — update "
                "params.json to match it, not the other way around."
            )


class TestNotebooksImportEngine:

    @pytest.mark.parametrize("notebook_name", CORE_NOTEBOOKS)
    def test_no_hardcoded_hyperparameters(self, notebook_name):
        path = REPO_ROOT / notebook_name
        assert path.exists(), f"missing {path}"
        nb = json.loads(path.read_text(encoding="utf-8"))

        offending = []
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for match in HYPERPARAM_PATTERN.finditer(src):
                offending.append(f"cell[{i}]: {match.group(0)}")

        assert not offending, (
            f"{notebook_name} has hardcoded hyperparameter-looking assignments "
            f"outside engine/: {offending}. Import from engine.train_models instead."
        )

    @pytest.mark.parametrize(
        "notebook_name",
        CORE_NOTEBOOKS + [
            f"exploratory/{n}" for n in [
                "FuelLatticeParameterModelCreation_TrainTest.ipynb",
                "FuelLatticeParameterNNModelCreation.ipynb",
                "FuelLatticeParameterSymbolicExpression.ipynb",
                "FuelLatticeParameterUTest.ipynb",
                "FuelLatticeParameterVisualization.ipynb",
            ]
        ],
    )
    def test_no_local_paths_or_revoked_key(self, notebook_name):
        path = REPO_ROOT / notebook_name
        assert path.exists(), f"missing {path}"
        text = path.read_text(encoding="utf-8")

        assert REVOKED_KEY not in text, (
            f"{notebook_name} still contains the revoked Materials Project API key prefix"
        )
        local_path_matches = LOCAL_PATH_PATTERN.findall(text)
        assert not local_path_matches, (
            f"{notebook_name} still contains a local machine path (C:/Users/... or C:\\Users\\...)"
        )
