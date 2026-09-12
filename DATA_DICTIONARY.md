# Data dictionary

Every column of every data file that this repository ships, with its meaning and
its units. Written because a table of numbers with bare column names is not
reusable data, and because review asked for descriptions of the variables rather
than only the values.

Lattice parameters are in ångström throughout, angles in degrees, and energies in
electronvolts per atom. Additionally, where a column can be missing, that is stated.

## Data/MP_Dataset_Original_Trimmed.csv

The Materials Project query result after symmetry annotation and removal of
noble-gas-containing compounds, 154,192 rows. This is the resume point that lets
the pipeline run without an API key, and it is the input to featurization rather
than to training: the 10 Å restriction and de-duplication that reduce it to the
64,128 training rows happen downstream.

| Column | Meaning |
|---|---|
| `material_id` | Materials Project identifier, e.g. `mp-1865`. Unique within a database version, and the key to trace any row back to the source. |
| `nsites` | Number of atomic sites in the conventional unit cell. Retained as a model feature. |
| `nelements` | Number of distinct chemical elements. Used for polymorph de-duplication, not as a feature. |
| `composition_reduced` | Reduced formula, e.g. `U1 N1`. Part of the de-duplication key. |
| `formation_energy_per_atom` | DFT formation energy, eV/atom. Used to pick which polymorph survives de-duplication, keeping the lowest. |
| `a`, `b`, `c` | Lattice parameters of the conventional cell, Å. These are the three prediction targets. Every value is DFT-relaxed at 0 K, which is the basis mismatch that Results/benchmarks/basis_check.md quantifies. |
| `alpha`, `beta`, `gamma` | Lattice angles, degrees. Carried through but not predicted. |
| `crystal_system` | One of cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic. Defines the cubic subset that the headline metrics are computed on. |
| `spacegroup_num` | International space group number, 1 to 230. Used as a scalar feature; the one-hot expansion of it is dropped before training. |
| `is_centrosymmetric` | Whether the space group contains an inversion centre. Boolean feature. |
| `n_symmetry_ops` | Order of the space group's symmetry operation set. Numeric feature. |

## Data/benchmarks/UNUC.csv

The U(N, C) validation system, 23 rows. `a_true` here is sampled from a quadratic
fit to literature measurements rather than being 23 independent experiments, which
matters when reading any correlation computed against it.

| Column | Meaning |
|---|---|
| `composition` | Solid-solution composition, e.g. `U1 N0.95 C0.05`. |
| `ref_mp-id` | Materials Project id of the reference host structure whose symmetry is fed to the model. An input, not a label. Resolved as the dominant end-member, breaking the 50/50 tie on formation energy; see Data/README.md. |
| `y` | Nitrogen fraction in UN(y)C(1-y), from 0 to 1. |
| `a_true` | Experimental lattice parameter, Å, read off the literature quadratic fit. |

## Data/benchmarks/CeO2Nd2O3Vals.csv

The (Ce, Nd)O2 validation system, 7 rows: six Nd-doped CeO2 samples measured for
this study, plus pure CeO2 from literature.

| Column | Meaning |
|---|---|
| `composition` | Composition, e.g. `Ce0.8343 Nd0.1657 O2`. The Nd fraction was determined after refinement from the lattice-parameter shift against the Ikuma et al. empirical fit, so it is a derived quantity rather than a weighed-in target. |
| `ref_mp-id` | Reference host, `mp-20194` (CeO2) on every row, since Ce is dominant throughout. |
| `a_true` | Lattice parameter, Å, from Rietveld refinement of the measured pattern in GSAS-II. The patterns themselves are held at ORNL; see AVAILABILITY.md. |

## Results/metrics/ModelMetrics_CrossVal.csv

Five-fold cross-validation aggregates, one row per model and lattice parameter,
so 15 rows for five models. Every value here is recomputable from the per-entry
predictions described below.

| Column | Meaning |
|---|---|
| `model_key` | `rf1`, `rf2`, `gbr1`, `gbr2` or `lin`. |
| `model_name` | Human-readable name, e.g. Lumped RF. |
| `param` | Which lattice parameter the row scores: `a`, `b` or `c`. |
| `MSE_all`, `MAE_all`, `R2_all` | Error over all crystal systems. Å² for MSE, Å for MAE, dimensionless for R². |
| `MSE_cubic`, `MAE_cubic`, `R2_cubic` | The same three restricted to cubic entries. These are the numbers the article quotes, because the fuels of interest are cubic and the all-systems figures are substantially worse. |

## Results/metrics/cv_predictions/cv_predictions_<key>.csv

The out-of-fold prediction for every training entry under one model, 64,128 rows
per file, five files. These are the points plotted in the predicted-versus-actual
figures, and every aggregate above is a reduction of them.

| Column | Meaning |
|---|---|
| `material_id` | Materials Project identifier, joinable to the source table. |
| `composition_reduced` | Reduced formula. |
| `crystal_system` | Crystal system of the entry. |
| `is_cubic` | Whether the entry is in the cubic subset the headline metrics use. |
| `model_key` | Which model produced the prediction. |
| `a_true`, `b_true`, `c_true` | Materials Project DFT lattice parameters, Å. |
| `a_pred`, `b_pred`, `c_pred` | Predicted lattice parameters, Å, from the fold in which this entry was held out. |

## Results/metrics/feature_spearman_cubic.csv

Spearman correlation between each model input feature and the true lattice
parameter, over cubic entries only. All 145 features appear, not only the ones
the figure shows.

| Column | Meaning |
|---|---|
| `n_cubic_rows` | Number of cubic rows the correlations were computed over. Constant down the file. |
| `feature` | Feature name, matching `Models/feature_labels/ML_FeatureLabels.json`. |
| `spearman_rho` | Spearman rank correlation with `a`, from -1 to 1. |
| `abs_rho` | Absolute value of the above, which the file is sorted by. |
| `in_figure` | Whether `abs_rho` exceeds 0.2, the display threshold used in the figure. The features below the threshold are kept because a weak correlation is a result too. |

## Results/benchmarks/<system>/predictions.csv

Per-composition predictions on a validation system, from every trained model.

| Column | Meaning |
|---|---|
| `composition` | The composition predicted. |
| `y` | Nitrogen fraction, present only in the U(N, C) file. |
| `a_true` | Experimental lattice parameter, Å. |
| `a_pred_rf1`, `a_pred_rf2`, `a_pred_gbr1`, `a_pred_gbr2` | Predicted lattice parameter, Å, per model. A cubic host collapses the model's three outputs to one reported value. |
| `a_pred_lin` | Linear Regression prediction, Å. Present only when the run passed `--include-baseline`, since the baseline is otherwise suppressed. |

## Results/benchmarks/basis_check.csv

Two kinds of row share this file, distinguished by `kind`, so most columns are
empty on any given row.

| Column | Meaning |
|---|---|
| `kind` | `anchor` for a DFT-versus-experiment comparison on one host, `metric` for a model's score on one benchmark. |
| `host`, `mp_id` | Anchor rows only: the host material and its Materials Project id. |
| `dft_a_angstrom`, `experimental_a_angstrom` | Anchor rows only: the two values being compared, Å. |
| `delta_dft_minus_exp_angstrom` | Anchor rows only: DFT minus experiment, Å. Its sign flips across the three anchors, which is why no correction factor is shipped. |
| `experimental_source` | Anchor rows only: which file and row the experimental value came from, or a statement that none exists in this repository. |
| `benchmark`, `system`, `n_rows` | Metric rows only: which validation set, and how many compositions. |
| `model_key`, `model_name` | Metric rows only: which model. |
| `MAE_vs_exp_angstrom` | Metric rows only: mean absolute error against the experimental value, Å. Contains the basis mismatch and is not pure model error. |
| `bias_angstrom` | Metric rows only: mean signed error, Å, i.e. the constant offset. |
| `scatter_angstrom` | Metric rows only: standard deviation of the error after removing the bias, Å. |
| `slope` | Metric rows only: slope of predicted against measured. Unaffected by a constant offset. |
| `pearson_r` | Metric rows only: correlation of predicted against measured. Also unaffected by a constant offset, and thus the defensible way to compare models across the basis mismatch. |

## Data/reference_systems.json

Curated symmetry and stability data for the nine end-member host structures the
predictor can resolve without a Materials Project API call. A JSON object keyed by
formula (`UN`, `UC`, `CeO2`, `UO2`, `PuO2`, `ThO2`, `ZrO2`, `Nd2O3`, `NdO2`), each
value an object with the fields below. Sourced from the Materials Project, CC BY 4.0.

| Field | Meaning |
|---|---|
| `mp_id` | Materials Project identifier for the host structure. |
| `crystal_system` | Crystal system of the host, e.g. `cubic`. Drives the collapse of a, b, c to a single reported `a`. |
| `spacegroup_num` | International space-group number, 1-230. Fed to the model as a scalar feature. |
| `is_centrosymmetric` | Whether the space group contains an inversion centre. Model feature. |
| `n_symmetry_ops` | Number of symmetry operations in the space group. Model feature. |
| `nsites` | Sites in the conventional cell. Model feature and dedup key. |
| `energy_above_hull` | eV/atom above the convex hull. Used only to break reference ties. |
| `formation_energy_per_atom` | eV/atom. The second tie-break, used when `energy_above_hull` cannot decide. |
| `note` | Free text recording how the entry was verified and any tie-break reasoning. Not consumed by code. |

## Results/benchmarks/benchmark_metrics.csv

Absolute-error summary for each model on each solid-solution benchmark. This file
reports error only; `basis_check.csv` carries the same models with the
offset-invariant slope and Pearson r alongside, and is what the manuscript's
benchmark table cites.

| Column | Meaning |
|---|---|
| `benchmark` | Benchmark key, `UNUC` or `CeO2Nd2O3`. |
| `system` | Human-readable system name, e.g. `U(N,C)`. |
| `n_rows` | Compositions scored, after dropping rows with no experimental value. |
| `model_key`, `model_name` | Which model the row describes. |
| `MAE_a_angstrom` | Mean absolute error of predicted `a` against the experimental value, Å. Contains the DFT-vs-experiment basis mismatch and is not pure model error. |
| `MSE_a_angstrom2` | Mean squared error of predicted `a`, Å². Same caveat. |

## Results/archive/*.csv

Three superseded optimization studies, retained for the record and **not for
citation**; `Results/README.md` explains what supersedes each. All three share the
quirk that `MSE`, `MAE` and `R2` hold a three-element array for a, b and c written
by numpy as a string, so parsing them takes more than a plain CSV read.

`CrystalSystemRandomForestRegressorOptimizationStudy.csv` and
`CrystalSystemRandomForestRegressorOptimizationStudy_Full.csv`:

| Column | Meaning |
|---|---|
| `lattice_parameter_threshold` | Å cut-off applied to a, b and c before fitting. |
| `OHE_method` | Which one-hot encoding was used, `sg` (space group) or `cs` (crystal system). |
| `crystal_system` | Crystal system the row's metrics are computed over. |
| `MSE`, `MAE`, `R2` | Three-element `[a b c]` arrays, as strings. |
| `dataset_size` | Rows surviving the threshold and dedup for that configuration. |

`RandomForestRegressorOptimizationStudy.csv` uses an earlier schema: an unnamed
integer index column, title-cased headers (`Lattice Parameter Threshold`,
`OHE Method`, `Dataset Size`), bracketed metric names (`MSE [a, b, c]`), and a
`cubic_only` boolean in place of the later `crystal_system` column.

## Results/CrystalSystemRandomForestRegressorOptimizationStudy_Final.csv

The per-crystal-system accuracy sweep for the Lumped RF model. Note that `MSE`,
`MAE` and `R2` here are strings holding a three-element array for a, b and c,
written by numpy rather than as separate columns, so parsing them takes more than
`float()`. That is awkward and was left as it is, because reformatting the file
would break its correspondence with the run that produced it.

| Column | Meaning |
|---|---|
| `lattice_parameter_threshold` | Upper bound on a, b and c for rows included, Å. |
| `OHE_method` | Which symmetry encoding the run used, `sg` for space group or `cs` for crystal system. |
| `crystal_system` | Which system the row scores, or `all`. |
| `MSE`, `MAE`, `R2` | Three-element arrays over a, b and c, in Å², Å and dimensionless. |
| `dataset_size` | Rows surviving the threshold and encoding for this configuration. |

This study uses a de-duplication key without `nsites`, which differs from the
training pipeline. Results/README.md explains why that difference is deliberate
and must not be reconciled.
