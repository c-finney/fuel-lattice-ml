"""
featurize.py — Shared featurization logic for build and predict.

Key guarantees:
  - make_featurizer() returns the canonical MultipleFeaturizer (same in build + predict)
  - base_feature_labels() returns exactly 503 labels with NO 'nelements'
    ('nelements' is used for dedup only, never a model feature — see the
    assert in base_feature_labels())
  - ohe_crystal_system / ohe_spacegroup use FIXED config.CS_LABELS / SG_LABELS,
    never re-derived from the data
  - featurize_compositions() uses ignore_errors=True and impute_nan=False defaults
"""

from __future__ import annotations

import pandas as pd

from engine import config


# ---------------------------------------------------------------------------
# Featurizer factory
# ---------------------------------------------------------------------------

def make_featurizer():
    """
    Return a configured MultipleFeaturizer matching the notebook exactly.

    Components (in order):
      ElementFraction, ElementProperty("magpie"), IonProperty(fast=True),
      ValenceOrbital(props=['avg']), Stoichiometry()

    Parallelism: n_jobs=4, chunksize=500 (matches notebook).

    impute_nan is left at the matminer default (False) — do NOT change this.
    Changing it shifts every Magpie feature value and invalidates the
    shipped model, which was trained (and validated) with this default.
    """
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementFraction,
        ElementProperty,
        IonProperty,
        Stoichiometry,
        ValenceOrbital,
    )

    ef = ElementFraction()
    ep = ElementProperty.from_preset("magpie")
    ip = IonProperty(fast=True)
    vo = ValenceOrbital(props=["avg"])
    st = Stoichiometry()

    featurizer = MultipleFeaturizer([ef, ep, ip, vo, st])
    featurizer.set_n_jobs(4)
    featurizer.set_chunksize(500)
    return featurizer


# ---------------------------------------------------------------------------
# OHE helpers — use FIXED label sets from config, never re-derived from the
# data. This guarantees a rebuild on a different Materials Project snapshot
# cannot silently reorder or drop a one-hot column that the shipped model
# expects at a fixed position.
# ---------------------------------------------------------------------------

def ohe_crystal_system(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode 'crystal_system' with prefix 'cs', reindex to
    config.CS_LABELS (fill missing with False).  Returns the OHE columns only.
    """
    dummies = pd.get_dummies(df["crystal_system"], prefix="cs")
    return dummies.reindex(columns=config.CS_LABELS, fill_value=False)


def ohe_spacegroup(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encode 'spacegroup_num' with prefix 'sg', reindex to
    config.SG_LABELS (fill missing with False).  Returns the OHE columns only.
    """
    dummies = pd.get_dummies(df["spacegroup_num"], prefix="sg")
    return dummies.reindex(columns=config.SG_LABELS, fill_value=False)


# ---------------------------------------------------------------------------
# Base feature label list (503 labels, NO nelements)
# ---------------------------------------------------------------------------

def base_feature_labels(featurizer) -> list[str]:
    """
    Return the 503-element feature label list used during build:
      ['spacegroup_num', 'is_centrosymmetric', 'n_symmetry_ops']
      + CS_LABELS (7)
      + SG_LABELS (230)
      + featurizer.feature_labels() (263)
      = 503 total

    'nelements' is intentionally absent: it is retained as a data column
    solely for training-time dedup (see engine/train_models.py), never as a
    model feature. If it were included here, predict_one would raise
    KeyError, because the prediction frame never constructs a 'nelements'
    column — the assert below guards exactly this.
    """
    symmetry_scalars = ["spacegroup_num", "is_centrosymmetric", "n_symmetry_ops"]
    labels = symmetry_scalars + config.CS_LABELS + config.SG_LABELS + featurizer.feature_labels()
    assert "nelements" not in labels, "nelements must NOT be in feature labels"
    return labels


# ---------------------------------------------------------------------------
# Featurization runner
# ---------------------------------------------------------------------------

def featurize_compositions(
    df: pd.DataFrame,
    comp_col: str,
    featurizer,
) -> pd.DataFrame:
    """
    Run featurizer.featurize_dataframe on *df* using *comp_col* as the
    composition column (must contain pymatgen Composition objects).

    ignore_errors=True keeps rows whose featurization partially fails,
    matching the original notebook's behaviour, rather than dropping them.
    Returns the augmented DataFrame.
    """
    return featurizer.featurize_dataframe(df, col_id=[comp_col], ignore_errors=True)
