"""Shared configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"

SAMPLE_RATE = 16_000  # 16 kHz mono
SEED = 42
MIN_SECONDS = 2.0
MAX_SECONDS = 30.0

MANIFEST_COLUMNS = [
    "file", "speaker_id", "label", "dataset", "language", "task",
    "sample_rate", "duration_s", "age", "gender", "sha1",
]  # label: 1 = PD, 0 = healthy

NOTICE = (
    "NeuroVoice is a screening-support research prototype. It is not a "
    "diagnosis. It was trained on small cohorts in three languages and has "
    "no clinical validation. Please talk to a doctor about any concerns."
)
