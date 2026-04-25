# deceptron/__init__.py
"""
Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion.

pip install deceptron   (once on PyPI)
pip install -e .        (from source)

from deceptron import DeceptronMLP, TrainConfig, train_forward, train_reverse,
                       train_reverse_jcp, SolverConfig, solve_dipg
"""
from .models  import DeceptronMLP, DeceptronCNN3D
from .jcp     import single_sample_probe_jcp, batch_probe_jcp, estimate_rjcp_dataset
from .train   import TrainConfig, train_forward, train_reverse, train_reverse_jcp
from .solvers import (SolverConfig,
                      solve_dipg,
                      solve_gradient_descent,
                      solve_gauss_newton,
                      solve_levenberg_marquardt)

__version__ = "1.0.0"
__author__  = "Aaditya L. Kachhadiya"

__all__ = [
    "DeceptronMLP", "DeceptronCNN3D",
    "single_sample_probe_jcp", "batch_probe_jcp", "estimate_rjcp_dataset",
    "TrainConfig", "train_forward", "train_reverse", "train_reverse_jcp",
    "SolverConfig", "solve_dipg",
    "solve_gradient_descent", "solve_gauss_newton", "solve_levenberg_marquardt",
]
