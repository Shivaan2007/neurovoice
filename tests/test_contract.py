"""The track-to-track contract from CLAUDE.md: meta.json keys and predict() output keys."""
import json
from pathlib import Path

from neurovoice.predict import predict

ROOT = Path(__file__).resolve().parent.parent


def test_meta_has_every_contract_key():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    for key in ("feature_names", "low", "high", "training_datasets", "model_version",
                "cross_dataset_headline", "limits"):
        assert key in meta, f"models/meta.json is missing contract key {key!r}"


def test_predict_stub_has_every_contract_key(tmp_path):
    out = predict("nothing.wav", model_path=tmp_path / "no.joblib", meta_path=tmp_path / "no.json")
    assert set(out) == {"probability", "band", "model_version", "stub", "notice"}
    assert out["stub"] is True
