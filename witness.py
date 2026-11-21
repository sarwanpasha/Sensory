#!/usr/bin/env python3
"""Constructed weights for the full-attention variant on XOR2.

Two temporary linear heads supervise the 2 comparison bits at their anchor
positions next to the usual label loss. The heads are then discarded and the
plain model is evaluated with its ordinary forward pass on fresh data.

  python witness.py --L 128
  python witness.py --L 256 --save full_xor2_L256.pt
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from leakgate import tasks as T
from models import DEVICE, TransformerClassifier, evaluate, n_params, onehot


def data(L, n, seed):
    rng = np.random.default_rng(seed)
    X, y, info = T.make_xor2(L, n, rng, T.model_anchors(L))
    return onehot(X), torch.as_tensor(y), torch.as_tensor(info["bits"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    (p1, _), (p2, _) = T.model_anchors(args.L)
    X, y, bits = data(args.L, args.n_train, args.seed)
    model = TransformerClassifier("full", args.L).to(DEVICE)
    aux = nn.ModuleList([nn.Linear(128, 2), nn.Linear(128, 2)]).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(aux.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    Xv, yv, _ = data(args.L, 4000, args.seed + 1)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            xb, yb, bb = X[idx].to(DEVICE), y[idx].to(DEVICE), bits[idx].to(DEVICE)
            h = model.states(xb)
            loss = (ce(model.head(h.mean(1)), yb)
                    + ce(aux[0](h[:, p1]), bb[:, 0]) + ce(aux[1](h[:, p2]), bb[:, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = evaluate(model, Xv, yv)
        print(f"epoch {epoch + 1:2d}  label AUROC {auc:.4f}")
        if auc >= 0.9999:
            break

    print(f"parameters of the plain model: {n_params(model)}")
    for seed in (555, 777):
        Xt, yt, _ = data(args.L, 6000, seed)
        print(f"fresh test set (seed {seed}): AUROC {evaluate(model, Xt, yt):.4f}")
    if args.save:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
