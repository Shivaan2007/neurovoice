"""Synthetic test audio: a speech-like signal that passes the speech check, and a tone."""
import io
import wave

import numpy as np

SR = 16000


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Harmonics on a gliding pitch, cut into ~4 syllables/s with short gaps and pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 130 + 30 * np.sin(2 * np.pi * 0.6 * t) + 15 * np.sin(2 * np.pi * 2.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    syllable = ((t * 4) % 1) < 0.7          # 175 ms voiced, 75 ms gap
    pause = (t % 2.5) > 2.1                 # 0.4 s pause every 2.5 s
    gate = syllable & ~pause
    y = np.where(gate, voice, 0.02 * rng.normal(size=t.size))
    return (0.3 * y / np.abs(y).max()).astype("float32")


def tone(seconds: float, hz: float = 150) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32")


def wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return buf.getvalue()
