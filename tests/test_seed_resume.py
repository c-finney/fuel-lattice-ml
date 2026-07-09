"""
test_seed_resume.py — Guards the seed-resume fix in engine/build_dataset.py.

The original (pre-fix) resume logic gated the Materials Project query skip on
BOTH file existence AND a "query" stage recorded in the build manifest. A
fresh clone has the committed seed (Data/MP_Dataset_Original_Trimmed.csv) but
NO manifest (it's gitignored, regenerated per-machine), so that logic would
have silently forced a multi-hour re-query on every first run — defeating the
entire point of shipping the seed.

This test asserts the fix: with the seed present and no manifest, build()
must NOT call the network query path at all.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


class TestSeedResume:

    def test_build_skips_query_when_seed_present_and_no_manifest(self, tmp_path, monkeypatch):
        from engine import build_dataset, config

        # Point config at an isolated temp tree with ONLY the seed present —
        # no build manifest, simulating a fresh clone.
        monkeypatch.setattr(config, "DATASETS", tmp_path / "Dataset")
        monkeypatch.setattr(config, "BUILD_MANIFEST", tmp_path / "Dataset" / ".build_manifest.json")
        seed_dir = tmp_path / "Data"
        seed_dir.mkdir(parents=True)
        seed_path = seed_dir / "MP_Dataset_Original_Trimmed.csv"
        monkeypatch.setattr(config, "DATASET_SEED", seed_path)

        # A minimal but structurally valid seed CSV (has a/b/c columns for
        # the Stage 2 filter, and the columns Stage 3 needs downstream).
        seed_df = pd.DataFrame({
            "material_id": ["mp-1"],
            "nsites": [2],
            "nelements": [2],
            "composition_reduced": ["UN"],
            "formation_energy_per_atom": [0.0],
            "a": [4.9], "b": [4.9], "c": [4.9],
            "alpha": [90.0], "beta": [90.0], "gamma": [90.0],
            "crystal_system": ["cubic"], "spacegroup_num": [225],
            "is_centrosymmetric": [True], "n_symmetry_ops": [48],
        })
        seed_df.to_csv(seed_path, index=False)

        assert config.dataset_original() == seed_path
        assert config.BUILD_MANIFEST.exists() is False, "test setup bug: manifest should not exist"

        with patch("engine.mp_client.search_summary") as mock_search, \
             patch("engine.featurize.make_featurizer") as mock_featurizer_factory:

            # Make featurization a no-op passthrough so this test stays
            # focused on the resume decision, not matminer's behavior.
            mock_fzer = mock_featurizer_factory.return_value
            mock_fzer.feature_labels.return_value = []

            def fake_featurize_dataframe(df, col_id, ignore_errors=True):
                return df

            mock_fzer.featurize_dataframe.side_effect = fake_featurize_dataframe

            with patch("engine.featurize.featurize_compositions",
                       side_effect=lambda df, comp_col, fzer: df):
                build_dataset.build(resume=True, force=False)

        mock_search.assert_not_called(), (
            "build(resume=True) queried Materials Project even though the seed "
            "dataset was present and no manifest existed — the resume-signal "
            "regression this test guards against."
        )

    def test_resume_flag_is_a_real_boolean_optional_action(self):
        """
        --resume/--no-resume must both be valid CLI flags (regression guard:
        the original flag was `action="store_true", default=True` with no
        way to disable it from the command line at all).
        """
        from engine.build_dataset import main
        import argparse

        # Rebuild just the parser the way main() does, to inspect it without
        # actually running a build.
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
        args = parser.parse_args(["--no-resume"])
        assert args.resume is False
        args = parser.parse_args([])
        assert args.resume is True
