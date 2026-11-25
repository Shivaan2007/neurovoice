"""The public Space README must quote the deployed model's current numbers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = (ROOT / "docs" / "space_README.md").read_text(encoding="utf-8")
META = json.loads((ROOT / "models" / "meta.json").read_text())


def test_space_readme_config_block():
    head = SPACE.split("---")[1]
    assert "sdk: docker" in head and "app_port: 7860" in head


def test_space_readme_numbers_match_model():
    cv, bp = META["cv"], META["band_precision"]
    for text in (f"{cv['auroc_window']:.3f} (95% CI {cv['window_ci95']})",
                 f"{cv['auroc_speaker']:.3f} (95% CI {cv['speaker_ci95']})",
                 f"about {round(bp['higher_crossfit'] * 100)}% of the time",
                 f"{META['n_speakers']} English speakers"):
        assert text in SPACE, f"Space README is missing or out of date: {text!r}"
