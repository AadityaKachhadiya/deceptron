"""
Optimization routines for Deceptron inverse problems.

This module provides D-IPG together with first- and second-order baseline
solvers. D-IPG uses both the forward surrogate ``f`` and the learned local
inverse ``g``. The baseline solvers use only the forward map, which keeps the
comparison focused on the contribution of the learned inverse.

The solvers operate on single flat input and observation tensors and return
standardized dictionaries containing the recovered estimate, iteration count,
success flag, residual, timing, and method-specific diagnostics.
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# Config 

@dataclass
class SolverConfig:
"""
Hyperparameters shared by the inverse-problem solvers.

Parameters control the iteration budget, stopping tolerance, Armijo
backtracking, domain projection, and method-specific step sizes or damping.
The defaults are conservative values intended for the included examples.
"""
    max_iterations: int        = 120
    tolerance_eps: float       = 0.30    # Heat-3D uses 0.38
    armijo_c: float            = 1e-4
    max_backtracking_steps:int = 8
    rho: float                 = 0.4
    x_low: float               = 0.0
    x_high: float              = 1.0
    dipg_step_size: float      = 1.0
    gd_step_size: float        = 1.0
    gn_step_size: float        = 1.0
    lm_step_size: float        = 1.0
    lm_damping: float          = 1e-2
    num_probes_eval: int       = 2       # for RJCP at final iterate


# Shared math helpers

def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _phi_grad(f, x: torch.Tensor, y_star: torch.Tensor):
"""
Evaluate the least-squares objective, gradient, and residual.

The objective is ``0.5 * mean((f(x) - y)^2)``. Returned tensors are detached
from the computation graph.
"""
    xr = x.detach().clone().requires_grad_(True)
    r  = f(xr) - y_star
    ph = 0.5 * torch.mean(r * r)
    g  = torch.autograd.grad(ph, xr)[0]
    return ph.detach(), g.detach(), r.detach()


def _phi_value(f, x: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    r = f(x) - y_star
    return 0.5 * torch.mean(r * r)


def _armijo_accept(f, x_t, p, y_star, rho, c, lo, hi):
"""
Evaluate one projected Armijo backtracking candidate.

The proposed direction ``p`` must have the same shape as ``x``. The returned
candidate is projected to the configured box constraints before evaluating
the sufficient-decrease condition.
"""
    phi_t, grad_t, _ = _phi_grad(f, x_t, y_star)
    x_trial   = _clamp((1.0 - rho) * x_t + rho * (x_t + p), lo, hi)
    phi_trial = _phi_value(f, x_trial, y_star)
    rhs       = phi_t + c * rho * torch.dot(grad_t.flatten(), p.flatten())
    return x_trial, phi_trial, rhs


def _normalized_residual(f, x: torch.Tensor, y_star: torch.Tensor) -> float:
    with torch.no_grad():
        r = f(x) - y_star
    return torch.sqrt(torch.mean(r * r)).item()


def _full_jacobian(f, x: torch.Tensor) -> torch.Tensor:
"""
Compute the Jacobian of a single-sample forward map.

The function is intended for moderate-dimensional problems where forming the
full Jacobian is practical.
"""
    xr = x.detach().clone().requires_grad_(True)
    def f_single(inp): return f(inp.unsqueeze(0)).squeeze(0)
    J = torch.autograd.functional.jacobian(f_single, xr, vectorize=True)
    return J.detach()


def _solve_linear(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
"""
Solve a dense linear system and return a one-dimensional solution tensor.

A least-squares fallback is used when the direct solve is ill-conditioned or
fails.
"""
    b = b.flatten()
    try:
        return torch.linalg.solve(A, b).flatten()
    except RuntimeError:
        return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1).flatten()


# D-IPG

def solve_dipg(
    model,
    y_star: torch.Tensor,
    x0: torch.Tensor,
    config: Optional[SolverConfig] = None,
) -> dict:
"""
Solve an inverse problem using D-IPG.

D-IPG forms proposal steps by applying the learned local inverse to residual
updates in observation space, followed by projected Armijo backtracking in
input space.

Parameters
----------
model : torch.nn.Module
    Trained module exposing ``model.f`` and ``model.g``.
y_star : torch.Tensor
    Target observation as a single flat tensor.
x0 : torch.Tensor
    Initial input estimate as a single flat tensor.
config : SolverConfig
    Solver hyperparameters.

Returns
-------
dict
    Dictionary containing the recovered estimate, iteration count, success
    flag, final residual, wall-clock time, and D-IPG diagnostics.
"""
    if config is None:
        config = SolverConfig()

    model.eval()
    device = next(model.parameters()).device
    x      = x0.clone().to(device).flatten()
    y_star = y_star.to(device).flatten()
    lo, hi = config.x_low, config.x_high

    acc_steps = 0
    res_drop_sum = alpha_sum = snorm_sum = cosine_sum = 0.0
    acc_count    = 0
    t0           = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, resid_t = _phi_grad(model.f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t ** 2)).item()
        if nr <= config.tolerance_eps:
            break

        alpha    = config.dipg_step_size
        accepted = False
        y_t      = model.f(x)

        for _ in range(config.max_backtracking_steps + 1):
            y_prop = y_t - alpha * resid_t
            x_prop = model.g(y_prop)
            p      = (x_prop - x).flatten()

            x_trial, phi_trial, rhs = _armijo_accept(
                model.f, x, p, y_star, config.rho, config.armijo_c, lo, hi)

            if phi_trial <= rhs:
                x_new = x_trial.detach().flatten()

                with torch.no_grad():
                    new_nr    = _normalized_residual(model.f, x_new, y_star)
                    res_drop_sum += (nr - new_nr)
                    sv = (x_new - x).flatten()
                    gv = grad_t.flatten()
                    sn = sv.norm().item()
                    gn = gv.norm().item()
                    cosine = (torch.dot(sv, -gv).item() / (sn * gn + 1e-12)
                              if sn > 0 and gn > 0 else 0.0)
                    alpha_sum  += alpha
                    snorm_sum  += sn
                    cosine_sum += cosine
                    acc_count  += 1

                x        = x_new
                accepted = True
                acc_steps += 1
                break

            alpha *= 0.5

        if not accepted:
            break

    elapsed     = time.perf_counter() - t0
    final_resid = _normalized_residual(model.f, x, y_star)

    # RJCP at final iterate
    from .jcp import single_sample_probe_jcp
    final_rjcp = float(
        single_sample_probe_jcp(
            model, x.detach().clone().requires_grad_(True),
            num_probes=config.num_probes_eval,
        ).detach().item()
    )

    return {
        "x_hat":                          x.detach().cpu(),
        "iters":                          iteration + 1,
        "success":                        float(final_resid <= config.tolerance_eps),
        "final_residual":                 final_resid,
        "accepted_frac":                  acc_steps / max(iteration + 1, 1),
        "time_sec":                       elapsed,
        "final_rjcp":                     final_rjcp,
        "mean_residual_drop_per_accept":  res_drop_sum / max(acc_count, 1),
        "mean_alpha_accept":              alpha_sum    / max(acc_count, 1),
        "mean_step_norm_accept":          snorm_sum    / max(acc_count, 1),
        "mean_cosine_with_neg_grad":      cosine_sum   / max(acc_count, 1),
    }


# Gradient Descent

def solve_gradient_descent(
    f,
    y_star: torch.Tensor,
    x0: torch.Tensor,
    config: Optional[SolverConfig] = None,
) -> dict:
"""
Solve an inverse problem with projected gradient descent.

The method uses only the forward map ``f`` and applies Armijo backtracking to
the negative gradient direction.
"""
    if config is None:
        config = SolverConfig()

    device = y_star.device
    x      = x0.clone().to(device).flatten()
    y_star = y_star.flatten()
    lo, hi = config.x_low, config.x_high
    acc_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, resid_t = _phi_grad(f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t ** 2)).item()
        if nr <= config.tolerance_eps:
            break

        eta, accepted = config.gd_step_size, False
        for _ in range(config.max_backtracking_steps + 1):
            p = -eta * grad_t
            x_trial, phi_trial, rhs = _armijo_accept(f, x, p, y_star,
                                                      config.rho, config.armijo_c, lo, hi)
            if phi_trial <= rhs:
                x = x_trial.detach().flatten()
                accepted = True; acc_steps += 1; break
            eta *= 0.5

        if not accepted:
            break

    return {
        "x_hat":         x.detach().cpu(),
        "iters":         iteration + 1,
        "success":       float(_normalized_residual(f, x, y_star) <= config.tolerance_eps),
        "final_residual": _normalized_residual(f, x, y_star),
        "accepted_frac": acc_steps / max(iteration + 1, 1),
        "time_sec":      time.perf_counter() - t0,
    }


# Gauss-Newton

def solve_gauss_newton(
    f,
    y_star: torch.Tensor,
    x0: torch.Tensor,
    config: Optional[SolverConfig] = None,
) -> dict:
"""
Solve an inverse problem with projected Gauss--Newton iterations.

The method uses only the forward map ``f``. It forms the local least-squares
linearization explicitly and applies Armijo backtracking to the resulting
step.
"""
    if config is None:
        config = SolverConfig()

    device = y_star.device
    x      = x0.clone().to(device).flatten()
    y_star = y_star.flatten()
    lo, hi = config.x_low, config.x_high
    acc_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, resid_t = _phi_grad(f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t ** 2)).item()
        if nr <= config.tolerance_eps:
            break

        J   = _full_jacobian(f, x)
        JTJ = J.T @ J / J.shape[0]
        JTr = J.T @ resid_t.flatten() / J.shape[0]
        dx  = _solve_linear(JTJ, -JTr)

        alpha, accepted = config.gn_step_size, False
        for _ in range(config.max_backtracking_steps + 1):
            p = (alpha * dx).flatten()
            x_trial, phi_trial, rhs = _armijo_accept(f, x, p, y_star,
                                                      config.rho, config.armijo_c, lo, hi)
            if phi_trial <= rhs:
                x = x_trial.detach().flatten()
                accepted = True; acc_steps += 1; break
            alpha *= 0.5

        if not accepted:
            break

    return {
        "x_hat":          x.detach().cpu(),
        "iters":          iteration + 1,
        "success":        float(_normalized_residual(f, x, y_star) <= config.tolerance_eps),
        "final_residual": _normalized_residual(f, x, y_star),
        "accepted_frac":  acc_steps / max(iteration + 1, 1),
        "time_sec":       time.perf_counter() - t0,
    }


# Levenberg-Marquardt

def solve_levenberg_marquardt(
    f,
    y_star: torch.Tensor,
    x0: torch.Tensor,
    config: Optional[SolverConfig] = None,
) -> dict:
"""
Solve an inverse problem with projected Levenberg--Marquardt iterations.

The method uses only the forward map ``f``. The Gauss--Newton normal equations
are damped by the configured Levenberg--Marquardt parameter before applying
projected Armijo backtracking.
"""
    if config is None:
        config = SolverConfig()

    device = y_star.device
    x      = x0.clone().to(device).flatten()
    y_star = y_star.flatten()
    lo, hi = config.x_low, config.x_high
    acc_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, resid_t = _phi_grad(f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t ** 2)).item()
        if nr <= config.tolerance_eps:
            break

        J   = _full_jacobian(f, x)
        JTJ = J.T @ J / J.shape[0]
        JTr = J.T @ resid_t.flatten() / J.shape[0]
        A   = JTJ + config.lm_damping * torch.eye(JTJ.shape[0], device=device)
        dx  = _solve_linear(A, -JTr)

        alpha, accepted = config.lm_step_size, False
        for _ in range(config.max_backtracking_steps + 1):
            p = (alpha * dx).flatten()
            x_trial, phi_trial, rhs = _armijo_accept(f, x, p, y_star,
                                                      config.rho, config.armijo_c, lo, hi)
            if phi_trial <= rhs:
                x = x_trial.detach().flatten()
                accepted = True; acc_steps += 1; break
            alpha *= 0.5

        if not accepted:
            break

    return {
        "x_hat":          x.detach().cpu(),
        "iters":          iteration + 1,
        "success":        float(_normalized_residual(f, x, y_star) <= config.tolerance_eps),
        "final_residual": _normalized_residual(f, x, y_star),
        "accepted_frac":  acc_steps / max(iteration + 1, 1),
        "time_sec":       time.perf_counter() - t0,
    }
