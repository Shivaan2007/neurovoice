"""Is this clip someone talking? A small, explainable guard in front of the model.

The model has only heard people reading aloud, so a beep, music or silence gets a
meaningless probability (a pure 150 Hz tone scored 0.85). This check rejects clips that
are clearly not speech before the model sees them, using three properties of talking:

* voiced_frac: share of frames with a detectable pitch. Speech mixes voiced sounds
  (vowels) with consonants and pauses; noise and silence are almost never voiced, and a
  steady tone, hum or chord is voiced all the time.
* pitch_spread: spread of the pitch, in semitones, over voiced frames. Talking moves the
  pitch up and down; a tone barely moves.
* loudness_variation: standard deviation of the loudness envelope divided by its mean.
  Syllables, consonants and pauses make speech loudness swing; tones, melodies played at
  an even volume, and steady noise stay flat.

Limits come from experiments/speech_check_calibration.py (results/speech_check.csv):
10 s clips from every MDVR-KCL recording (read and dialogue) and every Italian
read-text recording, PD and healthy, against synthetic tones, a melody, a chord, noise,
near-silence and hum. Each limit sits well clear of the lowest real-speech value. The
limits were chosen on that same set, so they are not an independent test. This guards
against honest mistakes, not against a determined attempt to fool it: a person singing
or a voice in music can pass.
"""
import numpy as np

from neurovoice.config import SAMPLE_RATE

HOP = 160  # 10 ms at 16 kHz
# Real speech, lowest / highest seen in calibration: voiced 0.13 / 0.93,
# pitch spread 1.29, loudness variation 0.455.
MIN_VOICED_FRAC = 0.08
MAX_VOICED_FRAC = 0.97
MIN_PITCH_SPREAD = 0.75  # semitones
MIN_LOUDNESS_VARIATION = 0.25


class NotSpeechError(ValueError):
    """Raised when a clip does not look like someone talking."""


def speech_stats(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    import librosa

    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, hop_length=HOP * 2)
    voiced_frac = float(np.mean(voiced)) if voiced.size else 0.0
    f0v = f0[voiced & ~np.isnan(f0)]
    pitch_spread = float(np.std(12 * np.log2(f0v / np.median(f0v)))) if f0v.size > 10 else 0.0
    env = librosa.feature.rms(y=y, frame_length=HOP * 4, hop_length=HOP)[0]
    loudness_variation = float(env.std() / env.mean()) if env.mean() > 0 else 0.0
    return {"voiced_frac": voiced_frac, "pitch_spread": pitch_spread,
            "loudness_variation": loudness_variation}


def failed_checks(stats: dict[str, float]) -> list[str]:
    reasons = []
    if stats["voiced_frac"] < MIN_VOICED_FRAC:
        reasons.append("almost no voice was detected (silence or noise?)")
    if stats["voiced_frac"] > MAX_VOICED_FRAC:
        reasons.append("the sound is continuous, with no pauses or consonants")
    if stats["pitch_spread"] < MIN_PITCH_SPREAD:
        reasons.append("the pitch barely moves, unlike talking")
    if stats["loudness_variation"] < MIN_LOUDNESS_VARIATION:
        reasons.append("the loudness stays flat, with no syllables or pauses")
    return reasons


def check_speech(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Return the stats if the clip looks like speech; raise NotSpeechError otherwise."""
    stats = speech_stats(y, sr)
    reasons = failed_checks(stats)
    if reasons:
        raise NotSpeechError("This doesn't sound like someone reading aloud: "
                             + "; ".join(reasons) + ". Please record yourself reading a "
                             "sentence or two in a quiet room.")
    return stats
