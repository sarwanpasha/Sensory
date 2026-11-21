#!/usr/bin/env python3
"""Numerical checks that need sampling.

  majority-sweep   window-fraction sweep for MAJORITY with a linear token-window probe
  gradient         pairing of the label with window-sum features (gradient identity)

  python numerical_checks.py majority-sweep --L 1280 --n 12000
  python numerical_checks.py gradient --L 128 --n 400000
"""
import argparse
from math import asin, pi, sqrt

import numpy as np

from leakgate import probes as P
from leakgate import tasks as T


def majority_sweep(args):
    rng = np.random.default_rng(args.seed)
    X, y, _ = T.make_majority(args.L, args.n, rng)
    print(f"{'f':>8}{'w':>6}{'AUROC':>9}{'adv':>9}{'A(f)':>9}")
    fs, adv = [], []
    for f in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
        w = int(f * args.L)
        auc = P.heldout_auroc(P.window(X, (args.L - w) // 2, w, 4), y, rng)
        fs.append(sqrt(f))
        adv.append(2 * auc - 1)
        print(f"{f:>8.5f}{w:>6}{auc:>9.4f}{2 * auc - 1:>9.4f}"
              f"{4 / pi * asin(sqrt(f / 2)):>9.4f}")
    slope, icpt = np.polyfit(fs, adv, 1)
    r2 = np.corrcoef(fs, adv)[0, 1] ** 2
    print(f"fit: advantage = {slope:.3f} sqrt(f) + {icpt:.3f}  (R^2 = {r2:.3f})")


def window_sum_features(X, half, rng, n_feat=16):
    """Random nonlinear features that are sums of functions of single windows."""
    n, L = X.shape
    E = np.eye(4, dtype=np.float32)[X]                        # n x L x 4
    feats = np.zeros((n, n_feat), dtype=np.float32)
    width = 2 * half + 1
    for j in range(n_feat):
        w = rng.normal(size=(width, 4)).astype(np.float32)
        bias = rng.normal()
        proj = np.zeros((n, L), dtype=np.float32)
        for d in range(-half, half + 1):
            lo, hi = max(0, -d), min(L, L - d)
            proj[:, lo:hi] += E[:, lo + d:hi + d] @ w[d + half]
        feats[:, j] = np.tanh(proj + bias).sum(1)
    return feats


def pairing(feats, y):
    ty = y - y.mean()
    f = feats - feats.mean(0)
    return np.abs(f.T @ ty) / (np.linalg.norm(f, axis=0) * np.linalg.norm(ty))


def gradient(args):
    rng = np.random.default_rng(args.seed)
    half = max(4, args.L // 32)      # reach of the 2-layer local variant
    noise = 1 / sqrt(args.n)
    print(f"noise scale 1/sqrt(n) = {noise:.2e}")
    cases = [
        ("xor2, separated", lambda: T.make_xor2(args.L, args.n, rng, T.model_anchors(args.L))),
        ("xor2, collocated", lambda: T.make_xor2(args.L, args.n, rng, T.collocated_anchors(args.L))),
        ("majority", lambda: T.make_majority(args.L, args.n, rng)),
    ]
    for name, gen in cases:
        X, y, _ = gen()
        c = pairing(window_sum_features(X, half, rng), y)
        print(f"{name:<18} max |corr| = {c.max():.2e}   median = {np.median(c):.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["majority-sweep", "gradient"])
    ap.add_argument("--L", type=int, default=1280)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    majority_sweep(args) if args.check == "majority-sweep" else gradient(args)


if __name__ == "__main__":
    main()
