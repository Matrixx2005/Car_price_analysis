"""Central configuration: paths, dataset source info, global constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLES = PROJECT_ROOT / "data" / "samples"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

RAW_FILENAME = "autoscout24_dataset_20251108.csv"
RAW_PATH = DATA_RAW / RAW_FILENAME

# Zenodo source (values below are the published ones; they are re-verified
# against the Zenodo API and the real file by download_dataset.py).
ZENODO_RECORD_ID = "17643343"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_DOI = "10.5281/zenodo.17643343"
ZENODO_FILE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{RAW_FILENAME}/content"
PUBLISHED_MD5 = "b23a122cc51baf7de39f449193ff0d28"
PUBLISHED_SIZE_BYTES = 548610318
DATASET_VERSION = "1.0.0"
DATASET_LICENSE = "MIT License (theo metadata Zenodo)"

RANDOM_SEED = 42
TEST_SIZE = 0.2
# Snapshot date encoded in the file name (autoscout24_dataset_20251108.csv).
SNAPSHOT_DATE = "2025-11-08"
SNAPSHOT_YEAR = 2025

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_SAMPLES, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)
