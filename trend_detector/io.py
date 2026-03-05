from pathlib import Path

# Project root = one level above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "DATA"


def get_data_path(filename):
    return DATA_DIR / filename