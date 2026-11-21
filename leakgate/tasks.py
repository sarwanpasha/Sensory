"""Synthetic sequence tasks built from balanced atoms.

Every generator returns (X, y, info) where X is an integer array of shape
(n, L) with tokens in {0, ..., alphabet-1}, y is a 0/1 label vector, and
info describes the positions the rule reads.
"""
import numpy as np


def _strict_pair(X, p, q, alphabet, rng):
    """Resample column q so that X[:, p] != X[:, q] (uniform over the rest)."""
    shift = rng.integers(1, alphabet, size=X.shape[0])
    X[:, q] = (X[:, p] + shift) % alphabet


def certified_anchors(L, rng):
    """Anchor bands of the certified XOR2 geometry: [L/8, L/4) and [3L/8, L/2)."""
    p1 = int(rng.integers(L // 8, max(L // 8 + 1, L // 4)))
    p2 = int(rng.integers(3 * L // 8, max(3 * L // 8 + 1, L // 2)))
    return (p1, L - 1 - p1), (p2, L - 1 - p2)


def model_anchors(L):
    """Fixed, well-separated geometry used in the transformer experiments."""
    return (L // 8, 5 * L // 8), (3 * L // 8, 7 * L // 8)


def collocated_anchors(L):
    """Adjacent anchors: a width-2 window exposes one coordinate of each atom."""
    return (2, L - 3), (3, L - 4)


def make_xor2(L, n, rng, pairs=None, alphabet=4):
    """Parity of 2 strict comparison atoms on disjoint supports."""
    if pairs is None:
        pairs = certified_anchors(L, rng)
    X = rng.integers(0, alphabet, size=(n, L))
    bits = []
    for p, q in pairs:
        _strict_pair(X, p, q, alphabet, rng)
        bits.append((X[:, p] < X[:, q]).astype(int))
    y = bits[0] ^ bits[1]
    return X, y, {"pairs": pairs, "bits": np.stack(bits, 1)}


def odd_pair_count(L):
    m = L // 2
    return m if m % 2 == 1 else m - 1


def make_majority(L, n, rng, alphabet=4):
    """Majority of m strict mirror-pair comparisons, m odd and close to L/2."""
    m = odd_pair_count(L)
    X = rng.integers(0, alphabet, size=(n, L))
    pairs = [(p, L - 1 - p) for p in range(m)]
    bits = np.empty((n, m), dtype=int)
    for i, (p, q) in enumerate(pairs):
        _strict_pair(X, p, q, alphabet, rng)
        bits[:, i] = X[:, p] < X[:, q]
    y = (bits.sum(1) * 2 > m).astype(int)
    return X, y, {"pairs": pairs, "bits": bits}


def threshold_triples(L):
    """Two disjoint, well-separated triples of positions."""
    a = (L // 8, 3 * L // 8, 5 * L // 8)
    b = (L // 4, L // 2, 7 * L // 8)
    return a, b


def make_threshold_parity(L, n, rng, triples=None, alphabet=6):
    """Parity of 2 threshold atoms b = 1[x_u + x_v + x_w >= 8] over 6 letters."""
    if triples is None:
        triples = threshold_triples(L)
    X = rng.integers(0, alphabet, size=(n, L))
    bits = [(X[:, list(t)].sum(1) >= 8).astype(int) for t in triples]
    y = bits[0] ^ bits[1]
    return X, y, {"triples": triples, "bits": np.stack(bits, 1)}


def make_first_occurrence(L, n, rng, alphabet=4):
    """Rejected candidate 1: y = 1 when letter 0 first appears before letter 1."""
    X = rng.integers(0, alphabet, size=(n, L))
    first0 = np.where((X == 0).any(1), (X == 0).argmax(1), L)
    first1 = np.where((X == 1).any(1), (X == 1).argmax(1), L)
    keep = first0 != first1
    X, first0, first1 = X[keep], first0[keep], first1[keep]
    return X, (first0 < first1).astype(int), {"accept": float(keep.mean())}


def make_fixed_comparison(L, n, rng, alphabet=4):
    """Rejected candidate 2: compare the tokens at L/4 and 3L/4."""
    X = rng.integers(0, alphabet, size=(n, L))
    p, q = L // 4, 3 * L // 4
    _strict_pair(X, p, q, alphabet, rng)
    y = (X[:, p] < X[:, q]).astype(int)
    return X, y, {"pairs": ((p, q),)}
