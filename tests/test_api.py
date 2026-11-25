from fastapi.testclient import TestClient

from api.main import app
from tests.synth import speechlike, tone, wav_bytes

client = TestClient(app)


def _wav(seconds):
    return wav_bytes(speechlike(seconds))


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


def test_tone_is_rejected_as_not_speech():
    r = client.post("/predict", files={"file": ("beep.wav", wav_bytes(tone(8)))})
    if r.status_code == 200 and r.json().get("stub"):
        return  # no model files: the stub does not run the check
    assert r.status_code == 422
    assert "doesn't sound like someone reading aloud" in r.json()["detail"]


def test_oversized_upload_rejected(monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 1000)
    assert client.post("/predict", files={"file": ("a.wav", _wav(6))}).status_code == 413
