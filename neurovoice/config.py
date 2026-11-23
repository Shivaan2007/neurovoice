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
MIN_SECONDS = 5.0  # matches the app's "5 to 20 seconds" instruction
MAX_SECONDS = 30.0
MAX_UPLOAD_BYTES = 20_000_000  # 20 MB is far more than 30 s of audio needs

MANIFEST_COLUMNS = [
    "file", "speaker_id", "label", "dataset", "language", "task",
    "sample_rate", "duration_s", "age", "gender", "sha1", "cohort",
]  # label: 1 = PD, 0 = healthy; cohort: e.g. young_hc / elderly_hc / pd (Italian only)

NOTICE = (
    "NeuroVoice is a screening-support research prototype. It is not a "
    "diagnosis. The model was trained on English speech from 37 people in one "
    "dataset, has not been tested on other languages, microphones or rooms, and "
    "has no clinical validation. Please talk to a doctor about any concerns."
)
