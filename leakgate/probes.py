"""Probes and feature views used by the leak gate."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def heldout_auroc(F, y, rng, test_frac=0.3):
    """Fit a logistic probe on a random split and return held-out AUROC."""
    n = len(y)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    F = np.asarray(F, dtype=float)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit((F[tr] - mu) / sd, y[tr])
    return float(roc_auc_score(y[te], clf.decision_function((F[te] - mu) / sd)))


def onehot(X, alphabet):
    """One-hot encode tokens and flatten over positions."""
    return np.eye(alphabet, dtype=np.float32)[X].reshape(X.shape[0], -1)


def histogram(X, alphabet):
    """Normalized token counts."""
    return np.stack([(X == a).mean(1) for a in range(alphabet)], 1)


def total_variation(H, y):
    """TV distance between the class-conditional mean histograms."""
    return 0.5 * float(np.abs(H[y == 1].mean(0) - H[y == 0].mean(0)).sum())


def window(X, start, width, alphabet):
    """One-hot view of a contiguous window."""
    start = int(np.clip(start, 0, X.shape[1] - width))
    return onehot(X[:, start:start + width], alphabet)
