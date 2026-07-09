"""
config.py — Single source of truth for paths, filenames, hyperparameters,
fixed label lists, MP query fields, and model registry.

All paths resolve from __file__ (never the cwd).
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]   # …/fuel-lattice-ml/

# Load .env BEFORE resolving DATA_ROOT below. load_mp_key() also calls
# load_dotenv(), but only lazily when a caller actually needs MP_API_KEY —
# by then this module has already finished executing top-level statements,
# so LATTICE_DATA_ROOT set only in .env (not a real process env var) would
# never be seen by the os.environ.get() call a few lines down. Loading here,
# unconditionally, at import time, is what makes .env-only configuration work
# for DATA_ROOT specifically. dotenv does not override a real env var that is
# already set, so an explicit process-level LATTICE_DATA_ROOT still wins.
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# Home for the big regenerable artifacts (model binary + generated datasets).
# Defaults to REPO_ROOT, so a fresh clone is fully self-contained. Only a
# consumer that wants to avoid a second multi-GB copy on disk (e.g. the
# fuel-agent MCP submodule, which points this back at the primary checkout)
# overrides it.
#
# `or REPO_ROOT` is load-bearing, NOT redundant: some launchers pass
# env = {"LATTICE_DATA_ROOT": "${LATTICE_DATA_ROOT}"}, and an unset variable
# expands to "". os.environ.get(k, default) returns "" (not the default) for
# a set-but-empty var, and Path("") == Path("."), which would silently
# redirect every artifact to the process's current working directory.
DATA_ROOT = Path(os.environ.get("LATTICE_DATA_ROOT") or REPO_ROOT)

DATA_DIR    = REPO_ROOT / "Data"                         # committed inputs
DATASETS    = DATA_ROOT / "Dataset"                      # generated, gitignored
MODELS_DIR  = DATA_ROOT / "Models" / "binaries"          # fetched, gitignored
FEATURE_DIR = REPO_ROOT / "Models" / "feature_labels"    # committed
METRICS_DIR = REPO_ROOT / "Results" / "metrics"          # committed
FIGURES_DIR = REPO_ROOT / "Results" / "figures"          # committed
CACHE_DIR   = DATASETS / "cache" / "ref_structures"      # regenerable, gitignored
REFERENCE_SYSTEMS = DATA_DIR / "reference_systems.json"  # committed — offline source of truth

BUILD_MANIFEST = DATASETS / ".build_manifest.json"       # regenerable, gitignored
MODEL_MANIFEST = REPO_ROOT / "Models" / "MANIFEST.json"  # committed provenance

# Ensure the regenerable artifact dirs exist at import time (safe no-op if they do)
for _d in [DATASETS, MODELS_DIR, METRICS_DIR, FIGURES_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thresholds (Å)
# ---------------------------------------------------------------------------
BUILD_THRESH = 20   # notebook 1: filter before featurization
TRAIN_THRESH = 10   # notebook 2: filter before training

# ---------------------------------------------------------------------------
# Fixed OHE label sets
#
# These are hardcoded here and never re-derived from the data (e.g. from
# df['crystal_system'].unique()), so a rebuild on a different Materials
# Project snapshot cannot silently reorder or drop a one-hot column that the
# shipped model expects at a fixed position.
# ---------------------------------------------------------------------------
CS_LABELS = [
    "cs_cubic", "cs_hexagonal", "cs_tetragonal", "cs_orthorhombic",
    "cs_monoclinic", "cs_triclinic", "cs_trigonal",
]  # 7 labels

SG_LABELS = [f"sg_{i}" for i in range(1, 231)]  # 230 labels

# ---------------------------------------------------------------------------
# MP query fields — only what downstream consumes.
#
# `structure` is transient: it is fed to SpacegroupAnalyzer for the lattice
# and symmetry fields, then dropped — never persisted to disk. That is why
# the committed seed dataset (Data/MP_Dataset_Original_Trimmed.csv) is 21 MB
# rather than gigabytes.
# ---------------------------------------------------------------------------
QUERY_FIELDS = [
    "material_id",
    "structure",               # transient: SpacegroupAnalyzer + lattice; NOT persisted
    "nsites",                  # raw doc.nsites — feature + dedup; see note in mp_client.py
    "elements",                # noble-gas filter (transient)
    "nelements",               # dedup only — deliberately NOT a model feature (see featurize.py)
    "composition_reduced",     # Composition() + dedup
    "formation_energy_per_atom",  # dedup sort key
]

# ---------------------------------------------------------------------------
# Model registry  key -> (filename, human_label)
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "rf1":  ("DependentRFModel.joblib",        "Dependent RF"),
    "rf2":  ("IndependentRFModel.joblib",       "Independent RF"),
    "gbr1": ("XGBoostGBRModel.joblib",          "Dependent GBR (XGBoost)"),
    "gbr2": ("ScikitLearnGBRModel.joblib",      "Independent GBR (HistGBR)"),
    "lin":  ("LinearRegressionModel.joblib",    "Linear Regression"),
}

# Linear Regression is trained only under `train --full` and is never shown
# in prediction output — it exists solely as a baseline sanity check.
REPORTABLE    = ["rf1", "rf2", "gbr1", "gbr2"]
# rf1 (Dependent RF) is currently the ONLY trained model shipped in this
# repository; the rest of this list is fallback order for if/when the others
# are trained. Do not describe this as an accuracy ranking — no comparative
# cross-validation across rf1/rf2/gbr1/gbr2 has been run. See
# Results/metrics/ModelMetrics_CrossVal.csv for rf1's own metrics.
HEADLINE_PREF = ["rf1", "rf2", "gbr2", "gbr1"]

# ---------------------------------------------------------------------------
# Artifact file paths (convenience)
# ---------------------------------------------------------------------------
DATASET_SEED       = DATA_DIR / "MP_Dataset_Original_Trimmed.csv"   # committed, plain blob
DATASET_FEATURIZED = DATASETS / "MP_Dataset_Featurized.csv"
DATASET_TRAINING   = DATASETS / "Training_Dataset.csv"
FEATURELABELS       = FEATURE_DIR / "FeatureLabels.joblib"
ML_FEATURELABELS    = FEATURE_DIR / "ML_FeatureLabels.joblib"

MODEL_URI_BASE = os.environ.get("LATTICE_MODEL_URI", "")   # e.g. hf://user/repo


def dataset_original() -> Path:
    """
    Return the locally regenerated trimmed CSV if a build has run, else the
    committed seed shipped in Data/. This is what lets a fresh clone skip the
    multi-hour Materials Project query — see build_dataset.py.
    """
    gen = DATASETS / "MP_Dataset_Original_Trimmed.csv"
    return gen if gen.exists() else DATASET_SEED


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_mp_key() -> str:
    """Load MP_API_KEY from .env; raise clearly if absent."""
    from dotenv import load_dotenv
    env_path = REPO_ROOT / ".env"
    load_dotenv(env_path)
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            f"MP_API_KEY is not set.\n"
            f"Create {env_path} with:\n"
            f"  MP_API_KEY=<your_materials_project_api_key>\n"
            f"Get a key at https://materialsproject.org/api"
        )
    return key


def model_path(key: str) -> Path:
    """Return the absolute path for a model key."""
    fname, _ = MODEL_FILES[key]
    return MODELS_DIR / fname


def model_exists(key: str) -> bool:
    """Return True if the model file for *key* exists on disk."""
    return model_path(key).exists()
