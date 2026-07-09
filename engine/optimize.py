"""
optimize.py — backs FuelLatticeParameterModelOptimization.ipynb (notebook 4).

The per-crystal-system / per-lattice-threshold optimization study that
produces Results/CrystalSystemRandomForestRegressorOptimizationStudy_Final.csv.

Dedup keys are deliberately configurable and NOT unified with
train_models.py's dedup: this study predates the addition of `nsites` to the
training pipeline's dedup key, and reproducing _Final.csv exactly requires
the 2-key variant. See Results/README.md for the full provenance discussion.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Reproduces Results/CrystalSystemRandomForestRegressorOptimizationStudy_Final.csv
DEDUP_KEYS_STUDY = ("composition_reduced", "spacegroup_num")
# Matches engine/train_models.py's prepare_training_frame()
DEDUP_KEYS_TRAIN = ("composition_reduced", "spacegroup_num", "nsites")


def dedup_dataframe(df: pd.DataFrame, dedup_keys: tuple[str, ...]) -> pd.DataFrame:
    """
    Dedup single-element polymorphs (keep lowest formation_energy_per_atom),
    using *dedup_keys* as the duplicate-detection subset.
    """
    df = df.sort_values("formation_energy_per_atom", ascending=True)
    nelements_mask = df["nelements"] == 1
    dupes_mask = df.duplicated(subset=list(dedup_keys), keep="first")
    mask = ~(nelements_mask & dupes_mask)
    df = df.loc[mask].reset_index(drop=True)
    df = df.sort_values("composition_reduced", ascending=True).reset_index(drop=True)
    return df


def evaluate_model(
    df_full: pd.DataFrame,
    feature_labels: list[str],
    sg_feature_labels: list[str],
    cs_feature_labels: list[str],
    lat_param_thresh: float,
    use_space_group_ohe: bool,
    crystal_system: str | None = None,
    dedup_keys: tuple[str, ...] = DEDUP_KEYS_TRAIN,
) -> dict:
    """
    Fit a single Dependent-RF-style model on a filtered slice of *df_full*
    and return its held-out MSE/MAE/R2 plus provenance fields.

    Parameters
    ----------
    df_full             : full featurized dataframe (post-dedup NOT required —
                           dedup is applied here via *dedup_keys*)
    feature_labels      : full feature label list (503, from FeatureLabels.joblib)
    sg_feature_labels   : the 230 sg_* one-hot column names
    cs_feature_labels   : the 7 cs_* one-hot column names
    lat_param_thresh    : filter rows to a,b,c <= this threshold (Å)
    use_space_group_ohe : True -> drop cs_* + scalar spacegroup_num, keep sg_*;
                           False -> drop sg_*, keep cs_* + scalar spacegroup_num
    crystal_system      : if set, restrict to this crystal system only
    dedup_keys          : passed to dedup_dataframe(); use DEDUP_KEYS_STUDY to
                           reproduce Results/..._Final.csv exactly, or
                           DEDUP_KEYS_TRAIN to match the training pipeline

    Returns
    -------
    dict with keys: lattice_parameter_threshold, OHE_method, crystal_system,
    MSE, MAE, R2, dataset_size
    """
    df = dedup_dataframe(df_full, dedup_keys)
    df = df[(df[["a", "b", "c"]] <= lat_param_thresh).all(axis=1)]

    if crystal_system is not None:
        df = df[df["crystal_system"] == crystal_system]

    if use_space_group_ohe:
        model_feature_labels = [c for c in feature_labels if c not in cs_feature_labels]
        model_feature_labels.remove("spacegroup_num")
    else:
        model_feature_labels = [c for c in feature_labels if c not in sg_feature_labels]

    # Drop columns with no values (all NaN, 0, or False) in this slice
    empty_cols_mask = ((df[model_feature_labels].isna()) | (df[model_feature_labels] == 0)).all()
    empty_cols = pd.Index(model_feature_labels)[empty_cols_mask].tolist()
    model_feature_labels = [c for c in model_feature_labels if c not in empty_cols]
    df = df.drop(columns=empty_cols)

    Y = df[["a", "b", "c"]]
    X = df[model_feature_labels]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=57)

    # Single random forest predicting a, b, c simultaneously (native
    # multi-output RandomForestRegressor — same family as the shipped rf1).
    model = RandomForestRegressor(n_estimators=600, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred, multioutput="raw_values")
    mae = mean_absolute_error(Y_test, Y_pred, multioutput="raw_values")
    r2  = r2_score(Y_test, Y_pred, multioutput="raw_values")

    return {
        "lattice_parameter_threshold": lat_param_thresh,
        "OHE_method": "sg" if use_space_group_ohe else "cs",
        "crystal_system": crystal_system,
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "dataset_size": len(df),
    }
