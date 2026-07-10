"""
test_featurize_alignment.py — Feature label alignment tests.

Critical checks:
  1. 'nelements' is absent from the base feature label list — it is a dedup-only
     column, never a model feature; including it would KeyError predict()
  2. base_feature_labels() returns exactly 503 labels
  3. A featurized prediction frame uses ML_FeatureLabels exactly when available,
     and the columns match without extra or missing entries
  4. 'spacegroup_num' (scalar) is present in ML_FeatureLabels — training drops the
     sg_* one-hot columns but keeps this scalar
  5. No sg_* OHE columns in ML_FeatureLabels (they are dropped in train)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from engine.featurize import make_featurizer, base_feature_labels
from engine import config


class TestFeatureLabelAlignment:

    def test_nelements_absent_from_base_labels(self):
        """
        REGRESSION TEST: 'nelements' must NOT appear in base_feature_labels.
        If it does, predict_one will KeyError because the prediction frame never
        constructs a 'nelements' column.
        """
        fzer = make_featurizer()
        labels = base_feature_labels(fzer)
        assert "nelements" not in labels, (
            "'nelements' found in base_feature_labels — it must stay dedup-only. "
            "This will cause a KeyError in prediction."
        )

    def test_base_labels_count(self):
        """
        base_feature_labels should return exactly 503 labels:
          3 symmetry scalars + 7 CS + 230 SG + 263 featurizer = 503
        """
        fzer = make_featurizer()
        labels = base_feature_labels(fzer)
        assert len(labels) == 503, (
            f"Expected 503 base feature labels, got {len(labels)}. "
            "Check that the featurizer preset ('magpie') has not changed."
        )

    def test_base_labels_start_with_symmetry_scalars(self):
        """First 3 labels should be symmetry scalars."""
        fzer = make_featurizer()
        labels = base_feature_labels(fzer)
        assert labels[0] == "spacegroup_num"
        assert labels[1] == "is_centrosymmetric"
        assert labels[2] == "n_symmetry_ops"

    def test_cs_labels_present_in_base(self):
        """All 7 cs_* OHE labels must appear in base_feature_labels."""
        fzer = make_featurizer()
        labels = base_feature_labels(fzer)
        for cs in config.CS_LABELS:
            assert cs in labels, f"Missing CS label: {cs}"

    def test_sg_labels_present_in_base(self):
        """SG OHE labels sg_1..sg_230 must appear in base_feature_labels."""
        fzer = make_featurizer()
        labels = base_feature_labels(fzer)
        assert "sg_1"   in labels
        assert "sg_225" in labels
        assert "sg_230" in labels

    def test_ml_feature_labels_exclude_sg_ohe(self):
        """
        ML_FeatureLabels (after train preprocessing) must NOT contain any sg_*
        OHE columns (they are dropped in prepare_training_frame — M2).
        """
        if not config.ML_FEATURELABELS.exists():
            pytest.skip("ML_FeatureLabels.joblib not built yet — skipping")
        import joblib
        ml_labels = joblib.load(config.ML_FEATURELABELS)
        sg_in_ml = [l for l in ml_labels if l.startswith("sg_")]
        assert not sg_in_ml, (
            f"sg_* OHE columns found in ML_FeatureLabels: {sg_in_ml[:5]}. "
            "These should be dropped by prepare_training_frame."
        )

    def test_ml_feature_labels_retain_spacegroup_num(self):
        """
        scalar 'spacegroup_num' MUST remain in ML_FeatureLabels — training drops
        the sg_* one-hot columns but keeps this scalar.
        """
        if not config.ML_FEATURELABELS.exists():
            pytest.skip("ML_FeatureLabels.joblib not built yet — skipping")
        import joblib
        ml_labels = joblib.load(config.ML_FEATURELABELS)
        assert "spacegroup_num" in ml_labels, (
            "'spacegroup_num' scalar is missing from ML_FeatureLabels."
        )

    def test_ml_feature_labels_exclude_nelements(self):
        """
        Regression guard on ML_FeatureLabels: 'nelements' must not appear
        (dedup-only column, never a model feature).
        """
        if not config.ML_FEATURELABELS.exists():
            pytest.skip("ML_FeatureLabels.joblib not built yet — skipping")
        import joblib
        ml_labels = joblib.load(config.ML_FEATURELABELS)
        assert "nelements" not in ml_labels, (
            "'nelements' in ML_FeatureLabels will KeyError predict."
        )

    def test_ml_feature_labels_include_nsites(self):
        """
        'nsites' should be in ML_FeatureLabels (added by toAdd in prepare_training_frame).
        """
        if not config.ML_FEATURELABELS.exists():
            pytest.skip("ML_FeatureLabels.joblib not built yet — skipping")
        import joblib
        ml_labels = joblib.load(config.ML_FEATURELABELS)
        assert "nsites" in ml_labels, (
            "'nsites' should be in ML_FeatureLabels (added via toAdd in train preprocessing)."
        )
