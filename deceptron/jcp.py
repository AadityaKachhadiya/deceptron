"""
deceptron/jcp.py
----------------
Jacobian Composition Penalty (JCP) — exact from the paper.

single_sample_probe_jcp  : per-sample RJCP estimate
batch_probe_jcp          : mean over a mini-batch (training loss term)
estimate_rjcp_dataset    : no-grad dataset diagnostic
"""

import torch
from torch.func import jvp


def single_sample_probe_jcp(
    model,
    x: torch.Tensor,
    num_probes: int = 2,
) -> torch.Tensor:
    """
    Estimate  E_xi[ ||J_g(f(x)) J_f(x) xi - xi||^2 ]  via Rademacher probes.

    Exact implementation from both Heat1D.py and Heat3D.py.

    Parameters
    ----------
    model : DeceptronMLP or DeceptronCNN3D
        Must have .f() and .g() accepting a single-sample (1D) tensor
        when wrapped with unsqueeze/squeeze.
    x : Tensor, shape (d,)
        Single input sample — NO batch dimension.
    num_probes : int
        Number of Hutchinson probes k.  Paper: k=2 (MLP train), k=4 (eval),
        k=1 (CNN train), k=2 (CNN eval).

    Returns
    -------
    Scalar tensor.
    """
    dim   = x.numel()
    total = x.new_tensor(0.0)

    def f_single(inp: torch.Tensor) -> torch.Tensor:
        return model.f(inp.unsqueeze(0)).squeeze(0)

    def g_single(inp: torch.Tensor) -> torch.Tensor:
        return model.g(inp.unsqueeze(0)).squeeze(0)

    for _ in range(num_probes):
        xi = torch.empty(dim, device=x.device).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        y  = f_single(x)
        _, jf_xi    = jvp(f_single, (x,), (xi,))
        _, jg_jf_xi = jvp(g_single, (y,), (jf_xi,))
        total = total + torch.mean((jg_jf_xi - xi) ** 2)

    return total / num_probes


def batch_probe_jcp(
    model,
    x_batch: torch.Tensor,
    num_probes: int = 2,
) -> torch.Tensor:
    """
    Mean RJCP over a mini-batch.  Used as the JCP training loss term.
    """
    values = [
        single_sample_probe_jcp(model, x_batch[i], num_probes=num_probes)
        for i in range(x_batch.shape[0])
    ]
    return torch.stack(values).mean()


@torch.no_grad()
def estimate_rjcp_dataset(
    model,
    x_data: torch.Tensor,
    num_probes: int = 4,
    max_samples: int = 64,
) -> float:
    """
    Runtime RJCP diagnostic — no gradients, no model.train().

    RJCP = 0  iff  J_g(f(x)) J_f(x) = I (perfect local inverse).
    Monitor during training: if RJCP plateaus above 1.0, train longer.

    Parameters
    ----------
    model : DeceptronMLP or DeceptronCNN3D
    x_data : Tensor — inputs to evaluate over (any device)
    num_probes : int — k=4 recommended for eval (optimal per paper Fig A4)
    max_samples : int — cap for speed

    Returns
    -------
    float : mean RJCP
    """
    model.eval()
    device = next(model.parameters()).device
    x_sub  = x_data[:max_samples].to(device)
    values = [
        single_sample_probe_jcp(model, x_sub[i], num_probes=num_probes).detach().cpu()
        for i in range(x_sub.shape[0])
    ]
    return float(torch.stack(values).mean().item())
