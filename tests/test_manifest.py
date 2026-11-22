import pandas as pd
import pytest

from neurovoice.manifest import check_one_label_per_speaker


def test_one_label_per_speaker_passes():
    df = pd.DataFrame({"speaker_id": ["a", "a", "b"], "label": [1, 1, 0]})
    check_one_label_per_speaker(df)


def test_speaker_with_two_labels_fails():
    df = pd.DataFrame({"speaker_id": ["a", "a", "b"], "label": [1, 0, 0]})
    with pytest.raises(ValueError, match="'a'"):
        check_one_label_per_speaker(df)
