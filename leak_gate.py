#!/usr/bin/env python3
"""Leak-gate certification of a synthetic sequence task.

For every length the script draws a fresh sample and runs 6 checks:
  G1  generator sanity (class balance)
  G2  frequency adversary (token histogram) and total variation
  G3  base-rate adversary (positive rate flat across lengths)
  G4  positive control (features the rule depends on)
  G4b anchor adversary (token at a single rule position)
  G5  locality adversary (contiguous window of width max(8, L/8))
The exit code is 0 only when every check passes at every length.

Examples:
  python leak_gate.py --task xor2
  python leak_gate.py --task majority --lgrid 256 512 640 1280
  python leak_gate.py --task threshold --lgrid 48 96 192
"""
import argparse
import sys

import numpy as np

from leakgate import probes as P
from leakgate import tasks as T

TAU_ADV, TAU_CTL = 0.55, 0.85
TV_MAX, BASE_RATE_SPREAD = 0.02, 0.02


def control_features(task, info):
    bits = info["bits"].astype(float)
    if task == "majority":
        return 2 * bits - 1
    return np.column_stack([bits, bits[:, 0] * bits[:, 1]])


def rule_positions(task, info):
    if task == "threshold":
        return [p for t in info["triples"] for p in t]
    return [p for pair in info["pairs"] for p in pair]


def run_length(task, L, n, rng):
    if task == "xor2":
        X, y, info = T.make_xor2(L, n, rng)
        alphabet = 4
    elif task == "majority":
        X, y, info = T.make_majority(L, n, rng)
        alphabet = 4
    else:
        X, y, info = T.make_threshold_parity(L, n, rng)
        alphabet = 6
    w = max(8, L // 8)
    H = P.histogram(X, alphabet)
    pos = rule_positions(task, info)
    anchor = max(P.heldout_auroc(P.onehot(X[:, [p]], alphabet), y, rng)
                 for p in pos[:4])
    # windows that are most favorable to a local reader
    starts = [pos[0] - w // 2, (L - w) // 2, 0]
    if task == "threshold":
        t = info["triples"][0]
        starts.append(t[0])
    locality = max(P.heldout_auroc(P.window(X, s, w, alphabet), y, rng)
                   for s in starts)
    return {
        "L": L, "w": w, "bal": float(y.mean()),
        "tv": P.total_variation(H, y),
        "freq": P.heldout_auroc(H, y, rng),
        "anchor": anchor, "locality": locality,
        "control": P.heldout_auroc(control_features(task, info), y, rng),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["xor2", "majority", "threshold"], default="xor2")
    ap.add_argument("--lgrid", type=int, nargs="+",
                    default=[20, 40, 80, 128, 256, 512, 640, 1280])
    ap.add_argument("--n", type=int, default=12000, help="sequences per length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [run_length(args.task, L, args.n, rng) for L in args.lgrid]

    print(f"{'L':>6}{'w':>6}{'bal':>8}{'TV':>9}{'FREQ':>8}{'ANCHOR':>8}"
          f"{'LOCAL':>8}{'CONTROL':>9}")
    for r in rows:
        print(f"{r['L']:>6}{r['w']:>6}{r['bal']:>8.3f}{r['tv']:>9.4f}"
              f"{r['freq']:>8.3f}{r['anchor']:>8.3f}{r['locality']:>8.3f}"
              f"{r['control']:>9.3f}")

    col = lambda k: np.array([r[k] for r in rows])
    gates = {
        "G1 generator sanity": bool(np.all(np.abs(col("bal") - 0.5) <= 0.05)),
        "G2 frequency adversary": bool(np.all(col("freq") <= TAU_ADV)
                                       and np.all(col("tv") <= TV_MAX)),
        "G3 base-rate adversary": bool(np.ptp(col("bal")) <= BASE_RATE_SPREAD),
        "G4 positive control": bool(np.all(col("control") >= TAU_CTL)),
        "G4b anchor adversary": bool(np.all(col("anchor") <= TAU_ADV)),
        "G5 locality adversary": bool(np.all(col("locality") <= TAU_ADV)),
    }
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(gates.values())
    print("CERTIFIED" if ok else "REJECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
