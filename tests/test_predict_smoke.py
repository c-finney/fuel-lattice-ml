"""
test_predict_smoke.py — Smoke tests for predict_one().

All heavy dependencies (MP, matminer, joblib models) are mocked.
Verifies:
  - UN0.5C0.5 returns status="ok", cubic, single-a headline
  - Headline value is in a physically sane range (~4.5-5.5 Å)
  - Table contains no Linear Regression row (LR is a baseline sanity check only,
    never shown in prediction output — see config.REPORTABLE)
  - The warnings field is present and is a list
  - Unknown model keys, and 'lin' without include_baseline, are reported as errors
  - An unparseable composition is reported as invalid_composition
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / mock helpers
# ---------------------------------------------------------------------------

_CUBIC_REF = {
    "status": "ok",
    "mp_id": "mp-1865",
    "crystal_system": "cubic",
    "spacegroup_num": 225,
    "is_centrosymmetric": True,
    "n_symmetry_ops": 48,
    "nsites": 2,
    "energy_above_hull": 0.0,
    "basis": "dominant mixed element (UN, fraction=0.5)",
    "source": "curated_table",
}


def _mock_prereqs_ok(stage: str) -> dict:
    return {
        "ok": True,
        "missing": [],
        "build_eta": "hours",
        "train_eta": "minutes",
        "available_models": ["rf1", "rf2"],
    }


def _make_mock_rf_model(a_val: float = 4.9):
    """Return a mock model whose predict() returns [[a,b,c]] for cubic."""
    model = MagicMock()
    # Cubic: a=b=c -> mean is a_val
    model.predict.return_value = np.array([[a_val, a_val, a_val]])
    return model


class TestPredictSmoke:

    @pytest.fixture(autouse=True)
    def _clear_predict_caches(self):
        """
        predict._load_model() and predict._load_ml_feature_labels() are
        functools.lru_cache-wrapped (see engine/predict.py). Without clearing
        them, a mocked joblib.load() from one test would leak its cached
        return value into every later test in this file, silently bypassing
        that test's own `patch("joblib.load", ...)` context.
        """
        from engine import predict
        predict._load_model.cache_clear()
        predict._load_ml_feature_labels.cache_clear()
        yield
        predict._load_model.cache_clear()
        predict._load_ml_feature_labels.cache_clear()

    def test_un05c05_cubic_headline(self):
        """
        UN0.5C0.5 should return:
          - status = "ok"
          - cubic headline with single 'a' key
          - a value in physically sane range [4.5, 5.5] Å
        """
        from engine import predict, config

        mock_labels = ["spacegroup_num", "is_centrosymmetric", "n_symmetry_ops",
                       "cs_cubic", "nsites"]  # minimal fake labels

        def mock_load(path, **kwargs):
            # **kwargs absorbs mmap_mode="r" — engine.predict._load_model()
            # always passes it (see engine/predict.py); a mock that only
            # accepted `path` would TypeError on every call.
            path = str(path)
            if "ML_FeatureLabels" in path:
                return mock_labels
            if "LumpedRF" in path:
                return _make_mock_rf_model(4.91)
            if "IndependentRF" in path:
                return _make_mock_rf_model(4.90)
            raise FileNotFoundError(f"Unexpected joblib.load call: {path}")

        # Mock featurization to return a DataFrame with the right columns
        import pandas as pd
        mock_X = pd.DataFrame([{l: 0.0 for l in mock_labels}])

        with patch("engine.artifacts.check_prereqs", side_effect=_mock_prereqs_ok), \
             patch("engine.predict.resolve_reference", return_value=_CUBIC_REF), \
             patch("engine.predict.build_prediction_frame", return_value=mock_X), \
             patch("joblib.load", side_effect=mock_load):

            result = predict.predict_one("UN0.5C0.5")

        assert result["status"] == "ok", f"Expected ok, got: {result}"

        hl = result["headline"]
        assert "a" in hl["values"], f"Cubic headline must have 'a' key, got: {hl['values']}"
        assert "b" not in hl["values"], "Cubic should collapse to single 'a', not a,b,c"
        assert hl["units"] == "Å"

        a_val = hl["values"]["a"]
        assert 4.5 <= a_val <= 5.5, f"Lattice parameter a={a_val:.4f} Å out of expected range [4.5, 5.5]"
        assert isinstance(result["warnings"], list)

    @staticmethod
    def _lr_fixture(lr_allowed: bool):
        """rf1 and the Linear Regression binary are both on disk; returns the patches."""
        import pandas as pd

        mock_labels = ["spacegroup_num", "is_centrosymmetric", "n_symmetry_ops",
                       "cs_cubic", "nsites"]

        def mock_load(path, **kwargs):
            path = str(path)
            if "ML_FeatureLabels" in path:
                return mock_labels
            if "LumpedRF" in path:
                return _make_mock_rf_model(4.91)
            if "LinearRegression" in path:
                if not lr_allowed:
                    raise AssertionError("Linear Regression model should not be loaded in predict!")
                return _make_mock_rf_model(5.80)
            raise FileNotFoundError(path)

        def mock_prereqs(stage):
            # artifacts.available_models() lists REPORTABLE keys only, so lin is absent
            # here even though its binary is present (model_exists is patched True).
            return {
                "ok": True,
                "missing": [],
                "build_eta": "hours",
                "train_eta": "minutes",
                "available_models": ["rf1"],
            }

        mock_X = pd.DataFrame([{l: 0.0 for l in mock_labels}])
        return [
            patch("engine.artifacts.check_prereqs", side_effect=mock_prereqs),
            patch("engine.config.model_exists", return_value=True),
            patch("engine.predict.resolve_reference", return_value=_CUBIC_REF),
            patch("engine.predict.build_prediction_frame", return_value=mock_X),
            patch("joblib.load", side_effect=mock_load),
        ]

    def _run(self, patches, **kwargs):
        from contextlib import ExitStack
        from engine import predict

        with ExitStack() as stack:
            for p_ in patches:
                stack.enter_context(p_)
            return predict.predict_one("UN0.5C0.5", **kwargs)

    def test_no_lr_in_table(self):
        """
        Table must not contain a Linear Regression row by default, even with its
        binary on disk: it is excluded from config.REPORTABLE and opted in only by
        include_baseline.
        """
        result = self._run(self._lr_fixture(lr_allowed=False))
        assert result["status"] == "ok"
        lr_rows = [r for r in result["table"] if r.get("model_key") == "lin"]
        assert not lr_rows, f"Linear Regression should not appear in table, but found: {lr_rows}"

    def test_lr_appears_only_with_include_baseline(self):
        """The same setup does produce a lin row once opted in, so the test above can fail."""
        result = self._run(self._lr_fixture(lr_allowed=True), include_baseline=True)
        assert result["status"] == "ok"
        assert [r for r in result["table"] if r["model_key"] == "lin"]

    def test_lin_requested_without_baseline_is_an_error(self):
        result = self._run(self._lr_fixture(lr_allowed=False), models=["rf1", "lin"])
        assert result["status"] == "error"
        assert "include-baseline" in result["reason"]

    def test_unknown_model_key_is_an_error(self):
        result = self._run(self._lr_fixture(lr_allowed=False), models=["foo"])
        assert result["status"] == "error"
        assert "foo" in result["reason"]

    def test_model_keys_are_case_insensitive(self):
        result = self._run(self._lr_fixture(lr_allowed=False), models=[" RF1"])
        assert result["status"] == "ok"
        assert [r["model_key"] for r in result["table"]] == ["rf1"]

    def test_invalid_composition(self):
        from engine import predict

        with patch("engine.artifacts.check_prereqs", side_effect=_mock_prereqs_ok):
            result = predict.predict_one("un0.5c0.5")
        assert result["status"] == "invalid_composition"

    def test_needs_reference_propagated(self):
        """
        If resolve_reference returns needs_reference, predict_one should propagate it.
        """
        from engine import predict

        needs_ref = {
            "status": "needs_reference",
            "candidates": ["UN", "UC"],
            "reason": "stoichiometric tie; no energy data",
        }

        mock_labels = ["spacegroup_num"]
        import pandas as pd
        mock_X = pd.DataFrame([{"spacegroup_num": 225}])

        with patch("engine.artifacts.check_prereqs", side_effect=_mock_prereqs_ok), \
             patch("engine.predict.resolve_reference", return_value=needs_ref), \
             patch("joblib.load", return_value=mock_labels):

            result = predict.predict_one("UN0.5C0.5")

        assert result["status"] == "needs_reference"

    def test_needs_build_when_no_models(self):
        """
        If no models are available, predict_one should return needs_build.
        """
        from engine import predict

        def mock_prereqs_no_models(stage):
            return {
                "ok": False,
                "missing": ["at least one trained model"],
                "build_eta": "hours",
                "train_eta": "minutes",
                "available_models": [],
            }

        with patch("engine.artifacts.check_prereqs", side_effect=mock_prereqs_no_models):
            result = predict.predict_one("UN0.5C0.5")

        assert result["status"] == "needs_build"


class TestPredictCsvNonCubicColumns:
    """
    Regression test for a real bug: predict_csv used to write
    result_row[f"a_pred_{model}"] unconditionally inside a loop over a/b/c,
    so for a non-cubic host every iteration overwrote the SAME column and
    the final value written (c) silently ended up under the "a_pred_*" name.
    Cubic hosts never exposed this because reporting.collapse() reduces them
    to a single "a" key before the loop runs — this test exercises the
    non-cubic path specifically.
    """

    def test_non_cubic_writes_distinct_abc_columns(self, tmp_path):
        from engine import predict

        non_cubic_ok = {
            "status": "ok",
            "headline": {
                "model": "rf1",
                "values": {"a": 3.5, "b": 3.5, "c": 6.0},
                "units": "Å",
            },
            "table": [
                {"model_key": "rf1", "model_name": "Lumped RF",
                 "a": "3.5000", "b": "3.5000", "c": "6.0000"},
            ],
            "reference": {"mp_id": "mp-2486", "crystal_system": "tetragonal"},
            "warnings": [],
        }

        csv_path = tmp_path / "non_cubic.csv"
        csv_path.write_text("composition\nU1N0.5C0.5\n", encoding="utf-8")

        with patch("engine.predict.predict_one", return_value=non_cubic_ok):
            result = predict.predict_csv(str(csv_path))

        df = result["dataframe"]
        assert "a_pred_rf1" in df.columns
        assert "b_pred_rf1" in df.columns
        assert "c_pred_rf1" in df.columns
        assert df.loc[0, "a_pred_rf1"] == 3.5
        assert df.loc[0, "b_pred_rf1"] == 3.5
        assert df.loc[0, "c_pred_rf1"] == 6.0
        # The bug this guards against: "a_pred_rf1" silently holding c's value.
        assert df.loc[0, "a_pred_rf1"] != df.loc[0, "c_pred_rf1"]
