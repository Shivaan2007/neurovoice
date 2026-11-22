"""Speaker-grouped vs naive splits."""
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from neurovoice.config import SEED


def grouped_splits(X, y, groups, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(X, y, groups))


def naive_splits(X, y, n_splits=5):
    """Ignores speakers. Leaks identity; use only for the leakage demo."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(X, y))
