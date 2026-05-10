# Deceptron

**Learned Local Inverses for Fast and Stable Physics Inversion.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PyTorch package for **D-IPG**, a learned local-inverse solver that amortizes inverse geometry for differentiable forward models and uses it to precondition iterative inverse-problem solves. Demonstrated on PDE inverse problems with higher reliability and substantially lower inference-time cost than classical baselines.



---

## Installation

**From source (recommended):**
```bash
git clone https://github.com/aadityakachhadiya/deceptron
cd deceptron
pip install -e .
```

**Direct install:**
```bash
pip install git+https://github.com/aadityakachhadiya/deceptron.git
```

Python ≥ 3.9, PyTorch ≥ 2.0.

---

## Quick start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from deceptron import (
    DeceptronMLP,
    TrainConfig, train_forward, train_reverse, train_reverse_jcp,
    SolverConfig, solve_dipg, estimate_rjcp_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. Define your model ─────────────────────────────────────
model = DeceptronMLP(dim=64).to(device)

# ── 2. Build data loaders ────────────────────────────────────
# y_train_norm must be z-score normalised forward outputs
tr_loader  = DataLoader(TensorDataset(x_train, y_train_norm), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val,   y_val_norm),   batch_size=128)

# ── 3. Train (3 stages, exact paper protocol) ────────────────
cfg = TrainConfig()   # paper defaults — see hyperparameter table below

model_s1   = train_forward(model, tr_loader, val_loader,
                            x_val.to(device), cfg, device)

model_s2   = train_reverse(model_s1, tr_loader, val_loader,
                            x_val.to(device), cfg, device)

model_plus = train_reverse_jcp(model_s2, tr_loader, val_loader,
                                x_val.to(device), cfg, device,
                                use_jcp=True,          # +JCP (recommended)
                                mlp_extra_losses=True) # MLP protocol

# ── 4. Inspect inverse quality ───────────────────────────────
rjcp = estimate_rjcp_dataset(model_plus, x_val, num_probes=4)
print(f"RJCP = {rjcp:.4f}  (< 1.0 = reliable, paper values: 0.18–0.66)")

# ── 5. Solve an inverse problem ──────────────────────────────
solver_cfg = SolverConfig(tolerance_eps=0.30, x_low=0.0, x_high=1.0)

result = solve_dipg(model_plus, y_observed.to(device),
                    x0=torch.zeros(64), config=solver_cfg)

print(f"success={result['success']}  iters={result['iters']}  "
      f"rjcp={result['final_rjcp']:.4f}")
print(result["x_hat"])
```

---

## Architecture options

### MLP — flat/vector inputs
For Heat-1D, Allen-Cahn (flat), Advection-Diffusion, Darcy, Navier-Stokes:
```python
model = DeceptronMLP(
    dim=64,                 # input = output dimension
    negative_slope=0.10,    # LeakyReLU slope
    hidden_multiplier=2,    # reverse map width = 2 * dim
)
```

### CNN — 3-D volumetric inputs
For Heat-3D and other spatial PDE problems:
```python
from deceptron import DeceptronCNN3D

model = DeceptronCNN3D(
    nx=8, ny=8, nz=8,       # spatial grid (input is flat: B × nx*ny*nz)
    hidden_channels=12,
    negative_slope=0.10,
)
```

---

## Hyperparameter guide

### TrainConfig defaults (MLP / Heat-1D)

| Parameter | MLP default | CNN (Heat-3D) | Notes |
|---|---|---|---|
| `forward_epochs` | 140 | 140 | |
| `reverse_epochs` | 100 | 100 | |
| `jcp_epochs` | 120 | 120 | |
| `cycle_weight` | 0.25 | **0.15** | |
| `probe_jcp_weight` | 1.0 | **0.35** | JCP loss scale |
| `jcp_num_probes_train` | 2 | **1** | k during training |
| `jcp_num_probes_eval` | 4 | **2** | k for RJCP eval |
| `jcp_batch_subsample` | 16 | **6** | batch cap for JCP |
| `y_tilde_noise` | 0.02 | **0.005** | cycle loss noise |
| `use_cosine_lr` | False | **True** | LR scheduler |

For CNN problems, override these in `TrainConfig` and pass `mlp_extra_losses=False`
to `train_reverse_jcp`.

### SolverConfig defaults

| Parameter | Default | Heat-3D |
|---|---|---|
| `tolerance_eps` | 0.30 | **0.38** |
| `max_iterations` | 120 | **80** |
| `dipg_step_size` | 1.0 | 1.0 |
| `x_low / x_high` | 0.0 / 1.0 | adjust to domain |

---

## Ablation mode (−JCP)

Reproduces the −JCP ablation from the paper:

```python
model_minus = train_reverse_jcp(
    model_s2, tr_loader, val_loader, x_val.to(device),
    cfg, device,
    use_jcp=False,           # -JCP: no Jacobian Composition Penalty
    mlp_extra_losses=True,
)

rjcp_p = estimate_rjcp_dataset(model_plus,  x_val)
rjcp_m = estimate_rjcp_dataset(model_minus, x_val)
print(f"+JCP RJCP = {rjcp_p:.4f}  (lower = better inverse)")
print(f"-JCP RJCP = {rjcp_m:.4f}")
print(f"Ratio     = {rjcp_m / rjcp_p:.2f}×  (paper: 1.4–3.6× depending on problem)")
```

---

## Benchmarks (validated on GPU, seed=7)

| Problem | D-IPG SR | Best baseline | Speedup vs GN |
|---|---|---|---|
| Heat-1D (MLP) | 24.3% | LM 35% | 19× (when GN works) |
| Heat-3D (CNN) | 100% | GN 29% | 19× |
| Allen-Cahn 2D | 100%* | L-BFGS 75% | 18× |

*Paper results with full GPU training. Package demos reproduce these with matching seeds.

---

## Package structure

```
deceptron/
├── deceptron/
│   ├── __init__.py    exports
│   ├── models.py      DeceptronMLP, DeceptronCNN3D
│   ├── jcp.py         JCP loss, RJCP diagnostic
│   ├── train.py       train_forward, train_reverse, train_reverse_jcp, TrainConfig
│   └── solvers.py     solve_dipg, solve_gd, solve_gn, solve_lm, SolverConfig
├── examples/
│   ├── heat1d_demo.py       Heat-1D (MLP) — validated ✓
│   ├── heat3d_demo.py       Heat-3D (CNN) — validated ✓
│   └── allen_cahn_demo.py   Allen-Cahn 2D — validated ✓
├── setup.py
└── README.md
```

---

## Citation

@article{kachhadiya2025deceptron,
  title   = {Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion},
  author  = {Kachhadiya, Aaditya L.},
  journal = {arXiv preprint arXiv:2511.21076},
  year    = {2025},
}

---

## License

MIT License. See [LICENSE](LICENSE) for details.
