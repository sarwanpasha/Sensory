# Leak-gate certification of long-range synthetic sequence tasks

Code for certifying that a synthetic sequence task is free of frequency,
base-rate, single-position, and bounded-window shortcuts, together with the
exact theory checks and the transformer experiments built on the certified
task.

## Contents

| File | Purpose |
|---|---|
| `leakgate/tasks.py` | Task generators: XOR2, MAJORITY, the threshold-atom family, and 2 rejected candidates |
| `leakgate/probes.py` | Logistic probes and feature views (histogram, single position, window) |
| `leak_gate.py` | The 6-gate certification protocol (CPU only) |
| `theory_checks.py` | Exact rational checks of the closed-form values and the arcsine law |
| `numerical_checks.py` | MAJORITY window-fraction sweep and the gradient-identity check |
| `models.py` | Transformer variants (`histogram`, `local`, `full`) and training |
| `witness.py` | Constructed full-attention weights for XOR2 |

## Setup

```bash
pip install -r requirements.txt
```

The certification and the theory checks need only NumPy, SciPy, and
scikit-learn. PyTorch is needed for `models.py` and `witness.py`.

## Usage

```bash
# certification
python leak_gate.py --task xor2
python leak_gate.py --task majority --lgrid 256 512 640 1280
python leak_gate.py --task threshold --lgrid 48 96 192

# exact values and sampling-based checks
python theory_checks.py
python numerical_checks.py majority-sweep --L 1280 --n 12000
python numerical_checks.py gradient --L 128 --n 400000

# trained variants on XOR2 and MAJORITY
python models.py --task xor2 --arm local --L 128 --epochs 80 --lr 2e-3 --seeds 6
python models.py --task majority --arm full --L 256 --epochs 40 --lr 1e-3 --patience 4

# from-scratch grid for the full variant (3 widths x 3 optimizer settings x 8 seeds)
for d in 128 256 512; do
  python models.py --task xor2 --arm full --L 128 --d-model $d --opt adamw --lr 2e-3 --seeds 8
  python models.py --task xor2 --arm full --L 128 --d-model $d --opt sgd --lr 0.1 --seeds 8
  python models.py --task xor2 --arm full --L 128 --d-model $d --opt sgd --lr 0.01 --seeds 8
done

# collocated contrast and constructed weights
python models.py --task xor2 --arm full --L 32 --geometry collocated --seeds 15
python witness.py --L 128
python witness.py --L 256
```

`leak_gate.py` exits with status 0 when every gate passes at every length and
with status 1 otherwise. Results that depend on sampling vary slightly with
the seed.

## License

See `LICENSE`.
