import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _wav(seconds):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        t = np.arange(int(16000 * seconds)) / 16000
        w.writeframes((np.sin(2 * np.pi * 150 * t) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


def test_health():
    assert client.get("/health").status_code == 200


def test_predict_has_fields_and_notice():
    r = client.post("/predict", files={"file": ("a.wav", _wav(6))})
    assert r.status_code == 200
    body = r.json()
    assert {"probability", "band", "model_version", "notice"} <= body.keys()
    assert body["notice"] and "verdict" not in body


def test_bad_file_rejected():
    assert client.post("/predict", files={"file": ("a.txt", b"x")}).status_code == 400
    assert client.post("/predict", files={"file": ("a.wav", _wav(3))}).status_code == 400


def test_oversized_upload_rejected(monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 1000)
    assert client.post("/predict", files={"file": ("a.wav", _wav(6))}).status_code == 413
