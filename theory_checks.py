#!/usr/bin/env python3
"""Exact checks of the closed-form values in the theory.

All enumerations use rational arithmetic, so the printed fractions are exact.

  python theory_checks.py
"""
from fractions import Fraction as F
from itertools import product
from math import asin, pi, sqrt

import numpy as np
from scipy.stats import binom


def auroc(cells):
    """Exact AUROC from a list of (score, label, probability)."""
    pos = [(s, p) for s, l, p in cells if l == 1]
    neg = [(s, p) for s, l, p in cells if l == 0]
    total = sum(p1 * p0 * (1 if s1 > s0 else F(1, 2) if s1 == s0 else 0)
                for s1, p1 in pos for s0, p0 in neg)
    return total / (sum(p for _, p in pos) * sum(p for _, p in neg))


def collect(pairs):
    out = {}
    for key, p in pairs:
        out[key] = out.get(key, 0) + p
    return [(s, l, p) for (s, l), p in out.items()]


def xor2_partial_exposure():
    """One coordinate of each comparison atom is visible (4 letters)."""
    m = lambda a: F(2 * a - 3, 3)
    cells = []
    for a1, c1, a2, c2 in product(range(4), repeat=4):
        if a1 == c1 or a2 == c2:
            continue
        y = int((a1 < c1) ^ (a2 < c2))
        cells.append(((-m(a1) * m(a2), y), F(1, 144)))
    rho = sum(m(a) ** 2 for a in range(4)) / 4
    return rho, rho ** 2, auroc(collect(cells))


def xor2_pooled_additive():
    """Best additive score g(x_p1) + h(x_p2) with tables in {-1, 0, 1}."""
    law = {}
    for a1, c1, a2, c2 in product(range(4), repeat=4):
        if a1 == c1 or a2 == c2:
            continue
        key = (a1, a2, int((a1 < c1) ^ (a2 < c2)))
        law[key] = law.get(key, 0) + F(1, 144)
    best = F(0)
    for g in product((-1, 0, 1), repeat=4):
        for h in product((-1, 0, 1), repeat=4):
            cells = collect(((g[a1] + h[a2], y), p) for (a1, a2, y), p in law.items())
            best = max(best, auroc(cells))
    return best


def threshold_family():
    """Threshold atoms b = 1[sum of 3 letters >= 8] over 6 letters."""
    def mu(known):
        vals = [(-1 if sum(known) + sum(r) >= 8 else 1)
                for r in product(range(6), repeat=3 - len(known))]
        return F(sum(vals), len(vals))

    mu1 = [mu((a,)) for a in range(6)]
    rho1 = sum(v * v for v in mu1) / 6
    rho2 = sum(mu((a, b)) ** 2 for a in range(6) for b in range(6)) / 36

    atom = {}
    for x in product(range(6), repeat=3):
        key = (x[0], int(sum(x) >= 8))
        atom[key] = atom.get(key, 0) + F(1, 216)

    parity = collect(((-mu1[a] * mu1[c], b1 ^ b2), p1 * p2)
                     for (a, b1), p1 in atom.items() for (c, b2), p2 in atom.items())
    maj_full = collect(((b[0], int(sum(b) >= 2)), F(1, 8))
                       for b in product((0, 1), repeat=3))
    cells = []
    for b1 in (0, 1):
        for (a, b2), p2 in atom.items():
            pb2 = (1 - mu1[a]) / 2
            post = sum((pb2 if u else 1 - pb2) * F(1, 2) * int(b1 + u + v >= 2)
                       for u in (0, 1) for v in (0, 1))
            for b3 in (0, 1):
                cells.append(((post, int(b1 + b2 + b3 >= 2)), p2 * F(1, 4)))
    return mu1, rho1, rho2, auroc(parity), auroc(maj_full), auroc(collect(cells))


def majority_window(m, f):
    """Exact AUROC and captured mass of the best reader of k = floor(f m) atoms."""
    k = int(f * m)
    a = np.arange(k + 1)
    pu, u = binom.pmf(a, k, 0.5), 2 * a - k
    b = np.arange(m - k + 1)
    pv, v = binom.pmf(b, m - k, 0.5), 2 * b - (m - k)
    ppos = np.array([pv[v > -x].sum() for x in u])
    mass = float((pu * (2 * ppos - 1) ** 2).sum())
    p1, p0 = 2 * pu * ppos, 2 * pu * (1 - ppos)
    auc = float((p1 * (np.cumsum(p0) - p0)).sum() + 0.5 * (p1 * p0).sum())
    return auc, mass


def main():
    rho, mass, auc = xor2_partial_exposure()
    print(f"XOR2, 1 coordinate per atom: rho={rho}, M={mass}, AUROC={auc}")
    print(f"XOR2, best additive pooled score over {{-1,0,1}} tables: AUROC={xor2_pooled_additive()}")
    mu1, rho1, rho2, par, majf, majp = threshold_family()
    print(f"Threshold atom: mu={[str(v) for v in mu1]}, rho1={rho1}, rho2={rho2}")
    print(f"  parity, 1 coordinate per atom: M={rho1 ** 2}, AUROC={par} ({float(par):.4f})")
    print(f"  majority of 3, 1 full atom: AUROC={majf}")
    print(f"  majority of 3, full atom + 1 coordinate: M={F(1, 4) + rho1 / 4}, "
          f"AUROC={majp} ({float(majp):.4f})")
    print("Majority, best atom-window reader at m=639 against the arcsine limit:")
    print(f"{'f':>8}{'AUROC':>9}{'limit':>9}{'mass':>9}{'limit':>9}")
    for f in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
        auc, mass = majority_window(639, f)
        print(f"{f:>8.5f}{auc:>9.4f}{0.5 + 2 / pi * asin(sqrt(f / 2)):>9.4f}"
              f"{mass:>9.4f}{2 / pi * asin(f):>9.4f}")


if __name__ == "__main__":
    main()
