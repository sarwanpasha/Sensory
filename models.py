#!/usr/bin/env python3
"""Transformer variants that differ only in receptive field, and their training.

Variants (same body, same parameter count):
  histogram  no positional encoding, mean pooling (token counts only)
  local      banded attention, each position attends within W/2, W = max(4, L/32)
  full       sinusoidal positional encoding and unrestricted attention

Examples:
  python models.py --task xor2 --arm local --L 128 --epochs 80 --lr 2e-3 --seeds 6
  python models.py --task majority --arm full --L 256 --epochs 40 --lr 1e-3 --patience 4
  # from-scratch grid cell: width 256, SGD with momentum
  python models.py --task xor2 --arm full --L 128 --d-model 256 --opt sgd --lr 0.1 \
      --epochs 80 --seeds 8
  # collocated contrast
  python models.py --task xor2 --arm full --L 32 --geometry collocated --seeds 15
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from leakgate import tasks as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHABET = 4


def sinusoidal(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class TransformerClassifier(nn.Module):
    def __init__(self, arm, L, d_model=128, heads=8, layers=2):
        super().__init__()
        self.arm = arm
        self.input_proj = nn.Linear(ALPHABET, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model,
                                           dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                  nn.Linear(d_model, 2))
        pe = sinusoidal(L, d_model)
        self.register_buffer("pe", torch.zeros_like(pe) if arm == "histogram" else pe)
        mask = None
        if arm == "local":
            half = max(4, L // 32) // 2
            idx = torch.arange(L)
            mask = (idx[None, :] - idx[:, None]).abs() > half
        self.register_buffer("mask", mask)

    def states(self, x):
        h = self.input_proj(x) + self.pe[: x.shape[1]].unsqueeze(0)
        return self.encoder(h, mask=self.mask)

    def forward(self, x):
        h = self.states(x)
        if self.arm == "local":
            return self.head(h).mean(1)
        return self.head(h.mean(1))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def onehot(X):
    return torch.nn.functional.one_hot(torch.as_tensor(X), ALPHABET).float()


def make_data(task, L, n, rng, geometry="separated"):
    if task == "majority":
        X, y, _ = T.make_majority(L, n, rng)
    else:
        pairs = T.model_anchors(L) if geometry == "separated" else T.collocated_anchors(L)
        X, y, _ = T.make_xor2(L, n, rng, pairs)
    return onehot(X), torch.as_tensor(y)


@torch.no_grad()
def evaluate(model, X, y, batch=512):
    model.eval()
    scores = [torch.softmax(model(X[i:i + batch].to(DEVICE)), 1)[:, 1].cpu()
              for i in range(0, len(X), batch)]
    return float(roc_auc_score(y.numpy(), torch.cat(scores).numpy()))


def train_one(args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr = make_data(args.task, args.L, args.n_train, rng, args.geometry)
    Xte, yte = make_data(args.task, args.L, args.n_test, rng, args.geometry)
    model = TransformerClassifier(args.arm, args.L, args.d_model).to(DEVICE)
    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    steps_per_epoch = math.ceil(len(Xtr) / args.batch)
    warm = max(1, args.warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = nn.CrossEntropyLoss()
    history, best, stale = [], 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss_fn(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE)).backward()
            opt.step()
            sched.step()
        auc = evaluate(model, Xte, yte)
        history.append(auc)
        stale = 0 if auc > best + 1e-4 else stale + 1
        best = max(best, auc)
        if args.patience and stale >= args.patience:
            break
    return {"seed": seed, "params": n_params(model), "final": history[-1],
            "max": max(history), "epochs": len(history)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["xor2", "majority"], default="xor2")
    ap.add_argument("--arm", choices=["histogram", "local", "full"], default="full")
    ap.add_argument("--geometry", choices=["separated", "collocated"], default="separated")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--opt", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=4, help="warmup epochs")
    ap.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--converged", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="optional JSON output file")
    args = ap.parse_args()

    runs = [train_one(args, s) for s in range(args.seeds)]
    for r in runs:
        print(json.dumps(r))
    finals = [r["final"] for r in runs]
    hits = sum(r["max"] >= args.converged for r in runs)
    print(f"mean AUROC {np.mean(finals):.4f}  range {min(finals):.4f}-{max(finals):.4f}  "
          f"converged {hits}/{len(runs)}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "runs": runs}, fh, indent=2)


if __name__ == "__main__":
    main()
