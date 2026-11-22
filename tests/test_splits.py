import numpy as np
from neurovoice.splits import grouped_splits


def test_no_speaker_in_train_and_test():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)
    y = np.repeat(rng.integers(0, 2, 40), 3)
    X = rng.normal(size=(120, 4))
    for tr, te in grouped_splits(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
