import numpy as np
import pytest

from neurovoice.speech_check import NotSpeechError, check_speech, failed_checks, speech_stats
from tests.synth import SR, speechlike, tone


def test_speechlike_signal_passes():
    stats = check_speech(speechlike(8))
    assert stats["voiced_frac"] < 0.97 and stats["loudness_variation"] > 0.25


@pytest.mark.parametrize("y", [
    tone(8),                                                        # steady beep
    np.random.default_rng(0).normal(size=SR * 8).astype("float32") * 0.1,  # noise
    np.random.default_rng(1).normal(size=SR * 8).astype("float32") * 1e-3,  # near-silence
])
def test_non_speech_is_rejected(y):
    with pytest.raises(NotSpeechError, match="doesn't sound like someone reading aloud"):
        check_speech(y)


def test_each_rule_names_its_reason():
    flat = {"voiced_frac": 1.0, "pitch_spread": 0.0, "loudness_variation": 0.01}
    reasons = failed_checks(flat)
    assert len(reasons) == 3 and any("pitch" in r for r in reasons)
    assert failed_checks({"voiced_frac": 0.6, "pitch_spread": 3.0,
                          "loudness_variation": 1.0}) == []


def test_stats_keys():
    assert set(speech_stats(speechlike(6))) == {"voiced_frac", "pitch_spread",
                                                "loudness_variation"}
