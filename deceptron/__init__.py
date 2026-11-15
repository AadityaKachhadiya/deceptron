# deceptron/__init__.py
"""Deceptron solvers, models, and training utilities."""
from .models  import DeceptronModule, DeceptronMLP, DeceptronCNN3D
from .jcp     import single_sample_probe_jcp, batch_probe_jcp, estimate_rjcp_dataset
from .train   import TrainConfig, train_forward, train_reverse, train_reverse_jcp
from .solvers import (SolverConfig,
                      solve_dipg,
                      solve_gradient_descent,
                      solve_gauss_newton,
                      solve_levenberg_marquardt)

__version__ = "1.0.0"
__author__  = "Aaditya L. Kachhadiya"

DIPG = solve_dipg

__all__ = [
    "DIPG", "DeceptronModule", "DeceptronMLP", "DeceptronCNN3D",
    "single_sample_probe_jcp", "batch_probe_jcp", "estimate_rjcp_dataset",
    "TrainConfig", "train_forward", "train_reverse", "train_reverse_jcp",
    "SolverConfig", "solve_dipg",
    "solve_gradient_descent", "solve_gauss_newton", "solve_levenberg_marquardt",
]
