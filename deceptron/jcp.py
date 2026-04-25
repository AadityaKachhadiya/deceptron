"""
Jacobian-composition utilities.

This module implements the Jacobian Composition Penalty (JCP) and the
corresponding runtime diagnostic (RJCP). For a forward map f and reverse map
g, the diagnostic estimates

    E_xi || J_g(f(x)) J_f(x) xi - xi ||^2,

using random Rademacher probes. Lower values indicate that g acts locally as
a better inverse of f around the sampled inputs.
"""

import torch
from torch.func import jvp


def single_sample_probe_jcp(
    model,
    x: torch.Tensor,
    num_probes: int = 2,
) -> torch.Tensor:
    """
    Estimate the Jacobian-composition error for one input sample.

    The model must provide two methods, ``f`` and ``g``. The estimate is

        E_xi || J_g(f(x)) J_f(x) xi - xi ||^2,

    where the expectation is approximated with ``num_probes`` Rademacher
    probes. The input ``x`` is expected to represent one sample without a
    batch dimension.

    Parameters
    ----------
    model : torch.nn.Module
        Module exposing ``model.f`` and ``model.g``.
    x : torch.Tensor
        Single input sample without a batch dimension.
    num_probes : int, default=1
        Number of random probes used in the estimate.

    Returns
    -------
    torch.Tensor
        Scalar JCP/RJCP estimate for the sample.
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
    Estimate the mean Jacobian-composition error over a mini-batch.

    This function applies ``single_sample_probe_jcp`` to samples from
    ``x_batch`` and averages the resulting scalar estimates. It can be used
    as a training loss term or as a small-batch diagnostic.

    Parameters
    ----------
    model : torch.nn.Module
        Module exposing ``model.f`` and ``model.g``.
    x_batch : torch.Tensor
        Batch of input samples.
    num_probes : int, default=1
        Number of random probes per sample.
    max_samples : int or None, default=None
        Optional cap on the number of batch samples used.

    Returns
    -------
    torch.Tensor
        Scalar mean JCP estimate over the selected samples.
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
    Estimate RJCP over a dataset subset.

    RJCP is a runtime diagnostic for the learned local inverse. It measures
    how close the composed Jacobian ``J_g(f(x)) J_f(x)`` is to the identity on
    randomly probed directions. The estimate is computed without updating the
    model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Module exposing ``model.f`` and ``model.g``.
    x_data : torch.Tensor
        Input samples used for the diagnostic.
    num_probes : int, default=2
        Number of random probes per sample.
    max_samples : int, default=64
        Maximum number of samples used from ``x_data``.

    Returns
    -------
    float
        Mean RJCP estimate over the selected samples.
    """
    model.eval()
    device = next(model.parameters()).device
    x_sub  = x_data[:max_samples].to(device)
    values = [
        single_sample_probe_jcp(model, x_sub[i], num_probes=num_probes).detach().cpu()
        for i in range(x_sub.shape[0])
    ]
    return float(torch.stack(values).mean().item())
