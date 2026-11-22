"""FastAPI backend. Never returns a yes/no verdict."""
import json
import os
import subprocess
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from neurovoice.config import (
    MAX_SECONDS,
    MAX_UPLOAD_BYTES,
    META_PATH,
    MIN_SECONDS,
    NOTICE,
    SAMPLE_RATE,
)
from neurovoice.predict import predict

app = FastAPI(title="NeuroVoice (research prototype)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return {"model_version": meta.get("model_version", "stub-0"), "meta": meta,
            "limits": NOTICE}


def _to_wav(src: str, dst: str) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", str(SAMPLE_RATE), dst],
                       capture_output=True)
    if r.returncode != 0:
        raise HTTPException(400, "Could not read this audio file.")


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so slow feature
# extraction does not block other requests.
@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".webm"}:
        raise HTTPException(400, "Unsupported format. Use wav, mp3, m4a or webm.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 1_000_000} MB).")
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            src = f.name
        dst = src + ".16k.wav"
        _to_wav(src, dst)
        import soundfile as sf
        info = sf.info(dst)
        dur = info.frames / info.samplerate
        if dur < MIN_SECONDS or dur > MAX_SECONDS:
            raise HTTPException(
                400, f"Clip must be {MIN_SECONDS:.0f} to {MAX_SECONDS:.0f} seconds long.")
        return predict(dst)
    finally:  # uploaded audio is deleted right after prediction
        for p in (src, dst):
            if p and os.path.exists(p):
                os.remove(p)
