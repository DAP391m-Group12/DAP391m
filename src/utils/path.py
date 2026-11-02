from pathlib import Path

# src/utils/path.py
# path.py -> utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT/ 'config'
DATA_DIR = PROJECT_ROOT/ 'data'
RAW_DATA_DIR = DATA_DIR /'raw'

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "model"
SQL_DIR = PROJECT_ROOT / "sql"
TESTS_DIR = PROJECT_ROOT / "tests"

# Local-only area, ignored by Git.
AGENT_DIR = PROJECT_ROOT / ".agent"
AGENT_TEST_RUNS_DIR = AGENT_DIR / "test-runs"

# Configuration entry points
PROJECT_CONFIG_PATH = CONFIG_DIR / "project.yaml"
CONFIG_CATALOG_PATH = CONFIG_DIR / "dataset_b" / "catalog.yaml"

# Current raw source
RAW_STATE_NY_PATH = RAW_DATA_DIR / "state_NY_2024.csv"

# Dataset outputs
DATASET_B_DIR = DATA_DIR / "dataset_b"

DATASET_DIRS = {
    "dataset_b": DATASET_B_DIR,
}

def processed_dataset_dir(dataset_name: str) -> Path:
    """Return the output directory for a registered research dataset."""
    try:
        return DATASET_DIRS[dataset_name]
    except KeyError as exc:
        allowed = ", ".join(DATASET_DIRS)
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. Allowed: {allowed}."
        ) from exc

