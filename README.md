# Deceptron

**D-IPG: Deceptron Inverse-Preconditioned Gradient for amortized local inverse geometry.**

**Paper:** *Local Inverse Geometry Can Be Amortized.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PyTorch package for **D-IPG** (*Deceptron Inverse-Preconditioned Gradient*), a learned local-inverse solver that amortizes local inverse geometry for differentiable forward models and uses it to precondition iterative inverse-problem solves.

Deceptron is designed for nonlinear inverse problems where many inverse instances share the same forward family, such as PDE inverse problems. Instead of repeatedly reconstructing local inverse geometry through Jacobian-based linear solves, Deceptron learns a reusable reverse operator whose differential action produces Gauss--Newton-like corrections at inference time.

---

## Core idea

Consider a differentiable forward surrogate

$$
f_W:\mathbb{R}^{d_{\mathrm{in}}}\to\mathbb{R}^{d_{\mathrm{out}}},
$$

where $x\\in\\mathbb\{R\}\^\{d\_\{\\mathrm\{in\}\}\}$ is the unknown latent state and $y\^\\star\\in\\mathbb\{R\}\^\{d\_\{\\mathrm\{out\}\}\}$ is the observed measurement. The inverse problem is posed as nonlinear least squares:

$$
\Phi(x)=\frac{1}{2}\|f_W(x)-y^\star\|_2^2.
$$

Classical nonlinear least-squares methods such as Gauss--Newton and Levenberg--Marquardt repeatedly reconstruct local inverse geometry by forming or applying the forward Jacobian $J\_f\(x\)$ and solving a linearized system at each iteration.

Deceptron learns a bidirectional module

$$
(f_W,g_V),
\qquad
g_V:\mathbb{R}^{d_{\mathrm{out}}}\to\mathbb{R}^{d_{\mathrm{in}}},
$$

where $f\_W$ is the learned forward surrogate and $g\_V$ is a learned reverse map. The reverse map is not used only as a one-shot inverse predictor. Instead, the goal is to train the **Jacobian of the reverse map** to behave like a local inverse of the forward Jacobian.

The central training signal is the **Jacobian Composition Penalty (JCP)**:

```math
\mathcal{L}_{\mathrm{JCP}}
=
\mathbb{E}_{x,\xi}
\left\|
J_g(f_W(x))J_f(x)\xi-\xi
\right\|_2^2,
```

where $\\xi$ is a random Hutchinson probe vector. Since

```math
\mathbb{E}_{\xi}\|A\xi\|_2^2=\|A\|_F^2,
```
JCP estimates the local composition defect

```math
J_g(f_W(x))J_f(x)-I
```

without explicitly forming full Jacobian matrices.

When this defect is small, the reverse Jacobian $J\_g\(f\_W\(x\)\)$ acts as a learned local left inverse of $J\_f\(x\)$. In full-column-rank least-squares regimes, this makes the induced D-IPG update locally Gauss--Newton-like.

---

## D-IPG inference update

At inference time, all model parameters are fixed and optimization is performed only over the latent variable $x$. Given the current iterate $x\_t$,

```math
y_t=f_W(x_t),
\qquad
r_t=y_t-y^\star.
```

D-IPG first takes a residual correction in measurement space:

```math
y_{t+1}^{\mathrm{prop}}=y_t-\alpha_t r_t.
```

It then pulls this measurement-space proposal back into latent space through the learned reverse map:

```math
x_{t+1}^{\mathrm{prop}}=g_V(y_{t+1}^{\mathrm{prop}}).
```

The proposal is relaxed, projected to feasible constraints when needed, and accepted using Armijo backtracking on the explicit objective $\\Phi\(x\)$.

A first-order expansion gives

```math
g_V(y_t-\alpha_t r_t)
=
g_V(y_t)-\alpha_t J_g(y_t)r_t
+
O(\alpha_t^2\|r_t\|_2^2).
```

When $g\_V\(f\_W\(x\_t\)\)\\approx x\_t$, this induces the latent-space step

```math
x_{t+1}^{\mathrm{prop}}
\approx
x_t-\alpha_t J_g(f_W(x_t))r_t.
```

Thus $J\_g\(f\_W\(x\_t\)\)$ acts as an amortized inverse-preconditioner. If

```math
J_g(f_W(x_t))\approx J_f(x_t)^+,
```

then D-IPG approximates a damped Gauss--Newton step

```math
x_t-\alpha_t J_f(x_t)^+r_t,
```

while avoiding a new Jacobian-based linear solve at every iteration.

---

## Why this is useful

D-IPG is intended for amortized inverse-problem regimes where:

- many inverse solves share the same forward family,
- classical Gauss--Newton or Levenberg--Marquardt solves are expensive,
- first-order methods are cheap but unreliable or slow,
- a learned reverse operator can be trained once and reused.

In this setting, Deceptron shifts part of the cost from repeated inference-time linear solves into one-time training of local inverse geometry.

The package includes:

- learned forward--reverse Deceptron modules,
- JCP training utilities,
- RJCP diagnostics for runtime inverse-consistency measurement,
- D-IPG inference solver,
- classical inverse-solver baselines such as GD, GN, LM, and L-BFGS-style comparisons in benchmark scripts.

---

## Installation

**From source:**
```bash
git clone https://github.com/AadityaKachhadiya/deceptron
cd deceptron
pip install -e .
```

**Direct install:**
```bash
pip install git+https://github.com/AadityaKachhadiya/deceptron.git
```

Python ≥ 3.9, PyTorch ≥ 2.0.

---

## Quick start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from deceptron import (
    DeceptronMLP,
    TrainConfig,
    train_forward,
    train_reverse,
    train_reverse_jcp,
    SolverConfig,
    solve_dipg,
    estimate_rjcp_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define a Deceptron model
model = DeceptronMLP(dim=64).to(device)

# 2. Build data loaders
# x_train: latent states
# y_train_norm: normalized forward observations
tr_loader = DataLoader(
    TensorDataset(x_train, y_train_norm),
    batch_size=128,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(x_val, y_val_norm),
    batch_size=128,
)

# 3. Train the forward surrogate
cfg = TrainConfig()

model_s1 = train_forward(
    model,
    tr_loader,
    val_loader,
    x_val.to(device),
    cfg,
    device,
)

# 4. Pretrain the reverse map
model_s2 = train_reverse(
    model_s1,
    tr_loader,
    val_loader,
    x_val.to(device),
    cfg,
    device,
)

# 5. Fine-tune the reverse map with JCP
model_plus = train_reverse_jcp(
    model_s2,
    tr_loader,
    val_loader,
    x_val.to(device),
    cfg,
    device,
    use_jcp=True,          # +JCP
    mlp_extra_losses=True, # MLP protocol
)

# 6. Inspect inverse-geometry quality 
rjcp = estimate_rjcp_dataset(model_plus, x_val, num_probes=4)
print(f"RJCP = {rjcp:.4f}")

# 7. Solve an inverse problem
solver_cfg = SolverConfig(
    tolerance_eps=0.30,
    max_iterations=120,
    dipg_step_size=1.0,
    x_low=0.0,
    x_high=1.0,
)

result = solve_dipg(
    model_plus,
    y_observed.to(device),
    x0=torch.zeros(64, device=device),
    config=solver_cfg,
)

print(f"success = {result['success']}")
print(f"iters   = {result['iters']}")
print(f"RJCP    = {result.get('final_rjcp', None)}")
print(result["x_hat"])
```

---

## Training protocol

Most experiments follow a three-stage protocol.

### Stage 1: forward surrogate training

Train $f\_W$ to approximate the forward map:

```math
\mathcal{L}_{\mathrm{task}}
=
\|f_W(x)-y\|_2^2.
```

### Stage 2: reverse-map pretraining

Freeze $f\_W$, then train $g\_V$ using reconstruction and cycle-consistency losses:

```math
\mathcal{L}_{\mathrm{rec}}
=
\|g_V(f_W(x))-x\|_2^2,
```

```math
\mathcal{L}_{\mathrm{cyc}}
=
\|f_W(g_V(\widetilde{y}))-\widetilde{y}\|_2^2.
```

### Stage 3: JCP fine-tuning

Initialize from the pretrained reverse map and continue training with or without JCP:

```math
\mathcal{L}
=
\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}
+
\lambda_{\mathrm{cyc}}\mathcal{L}_{\mathrm{cyc}}
+
\lambda_{\mathrm{JCP}}\mathcal{L}_{\mathrm{JCP}}
+
\mathcal{R}(W,V),
```

where $\\mathcal\{R\}\(W\,V\)$ denotes optional lightweight stabilization terms.

The $\+\\mathrm\{JCP\}$ and $\-\\mathrm\{JCP\}$ variants share the same forward model and the same reverse pretraining checkpoint. They differ only in the final reverse fine-tuning stage.

---

## Runtime diagnostic: RJCP

The same composition defect used for training can be evaluated at runtime:

```math
\mathrm{RJCP}(x)
=
\mathbb{E}_{\xi}
\left\|
J_g(f_W(x))J_f(x)\xi-\xi
\right\|_2^2.
```

RJCP measures whether the learned reverse geometry remains locally consistent along an optimization trajectory. Lower RJCP generally indicates better learned inverse consistency, although reliability also depends on conditioning, residual structure, and the quality of the forward surrogate.

```python
rjcp = estimate_rjcp_dataset(model_plus, x_val, num_probes=4)
print(f"RJCP = {rjcp:.4f}")
```

---

## Architecture options

### MLP -- flat/vector inputs

For vector-valued inverse problems:

```python
from deceptron import DeceptronMLP

model = DeceptronMLP(
    dim=64,
    hidden_multiplier=2,
    negative_slope=0.10,
)
```

### CNN -- spatial PDE inputs

For spatial PDE inverse problems:

```python
from deceptron import DeceptronCNN3D

model = DeceptronCNN3D(
    nx=8,
    ny=8,
    nz=8,
    hidden_channels=12,
    negative_slope=0.10,
)
```

---

## Hyperparameter guide

Representative training hyperparameters from the paper:

| Problem | S1 epochs | S2 epochs | S3 epochs | LR S1 | LR S2 | LR S3 | Cycle weight | JCP / probe weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heat-1D | 140 | 100 | 120 | 2e-3 | 2e-3 | 1e-3 | 0.25 | MLP stabilization |
| Heat-2D | 160 | 120 | 140 | 2e-3 | 2e-3 | 1e-3 | 0.15 | 0.50 |
| Heat-3D | 140 | 100 | 120 | 2e-3 | 2e-3 | 1e-3 | 0.15 | 0.35 |
| Darcy-2D | 140 | 60 | 80 | 2e-3 | 2e-3 | 1e-3 | 0.20 | 0.20 |
| Allen--Cahn-2D | 120 | 80 | 40 | 2e-3 | 1e-3 | 5e-4 | 0.05 | 0.01 |
| Adv.--Diff.-2D | 120 | 80 | 40 | 2e-3 | 1e-3 | 5e-4 | 0.05 | 0.01 |
| Navier--Stokes-2D | 120 | 80 | 15 | 2e-3 | 1e-3 | 5e-4 | 0.05 | 1e-3 |

S1 denotes forward-surrogate training, S2 denotes reverse-map pretraining, and S3 denotes reverse-map fine-tuning with JCP.

All listed runs use Adam with weight decay $10\^\{\-6\}$, reconstruction weight $1\.0$, and task weight $1\.0$. CNN-based benchmarks use cosine learning-rate schedules and gradient clipping. Heat-1D uses MLP-specific stabilization losses rather than the probe-JCP weight used in the CNN-based benchmarks.

### SolverConfig defaults

Representative D-IPG solver parameters:

| Parameter | Typical value | Meaning |
|---|---:|---|
| `dipg_step_size` | 1.0 | initial measurement-space residual correction scale |
| `tolerance_eps` | 0.30 | normalized residual tolerance |
| `max_iterations` | 80--120 | maximum D-IPG iterations |
| `armijo_c` | 1e-4 | Armijo sufficient-decrease constant |
| `relaxation` | 0.4 | relaxation applied before projection |
| `max_backtracking` | 8 | maximum Armijo backtracking steps |
| `x_low`, `x_high` | task-dependent | box constraints on latent state |

Example:

```python
solver_cfg = SolverConfig(
    tolerance_eps=0.30,
    max_iterations=120,
    dipg_step_size=1.0,
    armijo_c=1e-4,
    relaxation=0.4,
    max_backtracking=8,
    x_low=0.0,
    x_high=1.0,
)
```

---

## Ablation mode: −JCP

To evaluate the role of local inverse-geometry training, train a matched reverse model without the JCP term:

```python
model_minus = train_reverse_jcp(
    model_s2,
    tr_loader,
    val_loader,
    x_val.to(device),
    cfg,
    device,
    use_jcp=False,          # -JCP: no Jacobian Composition Penalty
    mlp_extra_losses=True,
)

rjcp_plus = estimate_rjcp_dataset(model_plus, x_val)
rjcp_minus = estimate_rjcp_dataset(model_minus, x_val)

print(f"+JCP RJCP = {rjcp_plus:.4f}")
print(f"-JCP RJCP = {rjcp_minus:.4f}")
print(f"Ratio     = {rjcp_minus / rjcp_plus:.2f}x")
```

This isolates whether the reverse map merely reconstructs states or also learns useful differential inverse structure.

---

## Benchmarks

The paper evaluates Deceptron/D-IPG on PDE inverse problems including:

- Heat-1D,
- Heat-2D,
- Heat-3D,
- Darcy-2D,
- Advection--Diffusion-2D,
- Allen--Cahn-2D,
- Navier--Stokes-2D.

Representative results:

| Problem | D-IPG(+JCP) success rate | Best classical baseline | Notes |
|---|---:|---|---|
| Heat-3D | 100% | LM | D-IPG matches recovery quality at much lower inference-time cost |
| Adv.-Diff.-2D | 100% | LM | D-IPG improves reliability and wall-clock solve time |
| Allen--Cahn-2D | 100% | L-BFGS | JCP strongly improves basin access and reliability |
| Darcy-2D | 68.8% | LM | hard ill-conditioned elliptic inverse problem |
| Heat-1D | 24.3% | GD / LM variants | known failure mode for learned reverse geometry |

The strongest results occur in amortized regimes where many inverse instances are solved for the same trained forward family.

---

## Inference-time cost intuition

Classical nonlinear least-squares methods repeatedly reconstruct local inverse geometry. For example, Gauss--Newton solves

```math
(J_f^\top J_f)\Delta x=-J_f^\top r
```

at each iteration, and Levenberg--Marquardt solves

```math
(J_f^\top J_f+\lambda I)\Delta x=-J_f^\top r.
```

D-IPG instead uses the learned reverse Jacobian action implicitly through $g\_V$. Once the Deceptron module is trained, each D-IPG proposal requires network evaluations and an Armijo acceptance check rather than a new Jacobian-based linear solve.

This does not make D-IPG free: Armijo acceptance can require additional objective and directional-derivative evaluations. However, the expensive local inverse operation is amortized into the trained reverse map.

---

## Package structure

```text
deceptron/
├── deceptron/
│   ├── __init__.py     exports
│   ├── models.py       DeceptronMLP, DeceptronCNN3D
│   ├── jcp.py          JCP loss and RJCP diagnostics
│   ├── train.py        train_forward, train_reverse, train_reverse_jcp, TrainConfig
│   └── solvers.py      solve_dipg, solve_gd, solve_gn, solve_lm, SolverConfig
├── examples/
│   ├── heat1d_demo.py
│   ├── heat3d_demo.py
│   └── allen_cahn_demo.py
├── setup.py
├── LICENSE
└── README.md
```

---

## Limitations

Deceptron depends on the quality of both the learned forward surrogate and the learned reverse geometry. If the forward model does not expose reliable local inverse structure, JCP cannot recover a trustworthy reverse differential operator by itself.

D-IPG is most useful in amortized settings. For one-off inverse problems, the training cost may outweigh the inference-time savings relative to classical methods.

Armijo backtracking is used as an acceptance safeguard for learned proposals. Because the proposal is generated through an amortized reverse map, the usual small-step line-search guarantee does not automatically apply in the same way as for classical descent directions.

---

## Citation

```bibtex
@article{kachhadiya2026localinversegeometry,
  title   = {Local Inverse Geometry Can Be Amortized},
  author  = {Kachhadiya, Aaditya L.},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

For the earlier workshop version:

```bibtex
@inproceedings{kachhadiya2025deceptron,
  title     = {Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion},
  author    = {Kachhadiya, Aaditya L.},
  booktitle = {NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences},
  year      = {2025},
  note      = {arXiv:2511.21076}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
