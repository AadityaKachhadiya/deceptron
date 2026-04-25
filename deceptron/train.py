"""
Training utilities for Deceptron models.

The training pipeline is organized into three stages. First, the forward map
is trained to approximate the observation operator. Second, the reverse map is
trained with reconstruction and cycle-consistency losses while the forward map
is frozen. Third, the reverse map can be fine-tuned with an optional
Jacobian-composition penalty while keeping the forward map fixed.

The functions in this module operate on models exposing ``f`` and ``g``
methods and return copies loaded with the best validation checkpoint.
"""

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .jcp import batch_probe_jcp, estimate_rjcp_dataset


# Config

@dataclass
class TrainConfig:
"""
Training hyperparameters for the three-stage Deceptron pipeline.

The configuration controls optimization budgets, learning rates, loss weights,
JCP probe counts, gradient clipping, validation subsampling, and scheduler
usage. Example scripts override these values for specific benchmark settings.
"""
    # Stage 1
    forward_epochs: int   = 140
    forward_lr: float     = 2e-3
    forward_wd: float     = 1e-6

    # Stage 2
    reverse_epochs: int   = 100
    reverse_lr: float     = 2e-3
    reverse_wd: float     = 1e-6

    # Stage 3
    jcp_epochs: int       = 120
    jcp_lr: float         = 1e-3
    jcp_wd: float         = 1e-6

    # Loss weights (MLP defaults)
    reconstruction_weight: float = 1.0
    cycle_weight: float          = 0.25   # Heat3D uses 0.15
    bias_tie_weight: float       = 5e-4   # optional for MLP only
    composition_weight: float    = 1e-3   # optional for MLP only
    probe_jcp_weight: float      = 1.0    # Heat3D uses 0.35

    # JCP settings
    jcp_num_probes_train: int = 2    # Heat3D uses 1
    jcp_num_probes_eval:  int = 4    # Heat3D uses 2
    jcp_batch_subsample:  int = 16   # Heat3D uses 6

    # Misc
    y_tilde_noise: float  = 0.02   # Heat3D uses 0.005
    gradient_clip: float  = 5.0
    eval_subset_rjcp: int = 64     # Heat3D uses 24

    # CNN-specific: set True to enable CosineAnnealingLR in all stages
    use_cosine_lr: bool = False    # Heat3D sets True


# Internal helpers

def _state_copy(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def _eval_forward_val(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
"""
Evaluate the forward-map validation loss over a full validation loader.

The returned value is the mean squared error between ``model.f(x)`` and the
target observation over all validation samples.
"""
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += float(F.mse_loss(model.f(xb), yb, reduction="sum").item())
        count += yb.numel()
    return total / max(count, 1)


# Stage 1
def train_forward(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_val: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
"""
Train the forward map.

Only ``model.forward_map`` is optimized in this stage. The reverse map is left
unchanged. The returned model is loaded with the checkpoint that achieves the
lowest forward validation loss.

Parameters
----------
model : torch.nn.Module
    Model exposing ``model.f`` and a ``forward_map`` parameter group.
train_loader : torch.utils.data.DataLoader
    Training batches of input--observation pairs.
val_loader : torch.utils.data.DataLoader
    Validation batches used for checkpoint selection.
x_val : torch.Tensor
    Validation inputs. Included for API consistency with later stages.
config : TrainConfig
    Training hyperparameters.
device : torch.device
    Device used for training.
verbose : bool, default=True
    Whether to print progress.

Returns
-------
torch.nn.Module
    Model loaded with the best validation checkpoint.
"""
    opt = torch.optim.Adam(
        model.forward_map.parameters(),
        lr=config.forward_lr,
        weight_decay=config.forward_wd,
    )
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.forward_epochs)
             if config.use_cosine_lr else None)

    best_val, best_state = float("inf"), None

    for epoch in range(config.forward_epochs):
        model.train()
        tr_tot, count = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.f(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.forward_map.parameters(), config.gradient_clip)
            opt.step()
            tr_tot += float(loss.item()) * xb.shape[0]
            count  += xb.shape[0]

        if sched is not None:
            sched.step()

        val_task = _eval_forward_val(model, val_loader, device)

        if val_task < best_val:
            best_val   = val_task
            best_state = _state_copy(model)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  [S1] epoch {epoch+1:03d} | "
                  f"train={tr_tot/count:.5f} | val={val_task:.5f}")

    model.load_state_dict(best_state)
    return model


# Stage 2

def train_reverse(
    model_stage1: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_val: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> nn.Module:
"""
Train the reverse map with the forward map frozen.

The reverse map is optimized using reconstruction and cycle-consistency
objectives. The input model is deep-copied before training, so the original
model is not modified. The returned model is loaded with the best validation
checkpoint according to the reverse-stage validation score.

Parameters
----------
model_stage1 : torch.nn.Module
    Model after forward-map training.
train_loader : torch.utils.data.DataLoader
    Training batches of input--observation pairs.
val_loader : torch.utils.data.DataLoader
    Validation batches used for checkpoint selection.
x_val : torch.Tensor
    Validation inputs used for RJCP diagnostics.
config : TrainConfig
    Training hyperparameters.
device : torch.device
    Device used for training.
verbose : bool, default=True
    Whether to print progress.

Returns
-------
torch.nn.Module
    Model loaded with the best reverse-stage checkpoint.
"""
    model = copy.deepcopy(model_stage1)
    _freeze(model.forward_map)

    opt = torch.optim.Adam(
        model.reverse_map.parameters(),
        lr=config.reverse_lr,
        weight_decay=config.reverse_wd,
    )
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.reverse_epochs)
             if config.use_cosine_lr else None)

    best_score, best_state = float("inf"), None

    for epoch in range(config.reverse_epochs):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                y_pred = model.f(xb)

            x_rec   = model.g(y_pred)
            y_tilde = yb + config.y_tilde_noise * torch.randn_like(yb)
            y_cyc   = model.f(model.g(y_tilde))

            loss = (config.reconstruction_weight * F.mse_loss(x_rec, xb) +
                    config.cycle_weight           * F.mse_loss(y_cyc, y_tilde))
            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            opt.step()

        if sched is not None:
            sched.step()

        val_task = _eval_forward_val(model, val_loader, device)
        val_rjcp = estimate_rjcp_dataset(model, x_val,
                                          config.jcp_num_probes_eval,
                                          config.eval_subset_rjcp)
        score = val_rjcp + 1e-3 * val_task

        if score < best_score:
            best_score = score
            best_state = _state_copy(model)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  [S2] epoch {epoch+1:03d} | val={val_task:.5f} | rjcp={val_rjcp:.5f}")

    model.load_state_dict(best_state)
    return model


# Stage 3

def train_reverse_jcp(
    model_stage2: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_val: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    use_jcp: bool = True,
    mlp_extra_losses: bool = True,
    verbose: bool = True,
) -> nn.Module:
"""
Fine-tune the reverse map with an optional Jacobian-composition penalty.

The forward map remains frozen. The input model is deep-copied before
fine-tuning, allowing JCP and non-JCP variants to start from the same reverse
checkpoint. When ``use_jcp`` is enabled, the objective includes a probe-based
penalty that encourages ``J_g(f(x)) J_f(x)`` to approximate the identity.

Parameters
----------
model_stage2 : torch.nn.Module
    Model after reverse-map training.
train_loader : torch.utils.data.DataLoader
    Training batches of input--observation pairs.
val_loader : torch.utils.data.DataLoader
    Validation batches used for checkpoint selection.
x_val : torch.Tensor
    Validation inputs used for RJCP diagnostics.
config : TrainConfig
    Training hyperparameters.
device : torch.device
    Device used for training.
use_jcp : bool, default=True
    Whether to include the Jacobian-composition penalty.
mlp_extra_losses : bool, default=False
    Whether to include optional MLP-specific auxiliary losses.
verbose : bool, default=True
    Whether to print progress.

Returns
-------
torch.nn.Module
    Model loaded with the best fine-tuning checkpoint.
"""
    model = copy.deepcopy(model_stage2)
    _freeze(model.forward_map)

    opt = torch.optim.Adam(
        model.reverse_map.parameters(),
        lr=config.jcp_lr,
        weight_decay=config.jcp_wd,
    )
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.jcp_epochs)
             if config.use_cosine_lr else None)

    # Pre-compute identity for composition loss (MLP only)
    if mlp_extra_losses:
        dim      = model.forward_map.linear.weight.shape[0]
        identity = torch.eye(dim, device=device)

    best_score, best_state = float("inf"), None

    for epoch in range(config.jcp_epochs):
        model.train()
        tot = dict(loss=0., rec=0., cyc=0., bias=0., comp=0., jcp=0.)
        count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            # f is frozen, detach to avoid building the graph through f
            with torch.no_grad():
                y_pred = model.f(xb)

            x_rec   = model.g(y_pred)
            y_tilde = yb + config.y_tilde_noise * torch.randn_like(yb)
            y_cyc   = model.f(model.g(y_tilde))

            loss_rec = F.mse_loss(x_rec, xb)
            loss_cyc = F.mse_loss(y_cyc, y_tilde)

            loss = (config.reconstruction_weight * loss_rec +
                    config.cycle_weight           * loss_cyc)

            if mlp_extra_losses:
                # bias tie  (Heat-1D protocol)
                fwd_bias = model.forward_map.linear.bias.detach()
                rev_bias = model.reverse_map.network[0].bias
                loss_bias = torch.mean((fwd_bias.mean() + rev_bias.mean()) ** 2)

                # composition  V[:d,:] @ W ≈ I  (Heat-1D protocol)
                fwd_W  = model.forward_map.linear.weight.detach()
                rev_W0 = model.reverse_map.network[0].weight[:, :dim]
                comp   = rev_W0[:dim, :] @ fwd_W
                loss_comp = torch.mean((comp - identity) ** 2)

                loss = (loss +
                        config.bias_tie_weight   * loss_bias +
                        config.composition_weight * loss_comp)

                tot["bias"] += float(loss_bias.item()) * xb.shape[0]
                tot["comp"] += float(loss_comp.item()) * xb.shape[0]

            if use_jcp:
                n_sub    = min(config.jcp_batch_subsample, xb.shape[0])
                loss_jcp = config.probe_jcp_weight * batch_probe_jcp(
                    model, xb[:n_sub], num_probes=config.jcp_num_probes_train)
                loss = loss + loss_jcp
                tot["jcp"] += float(loss_jcp.item()) * xb.shape[0]

            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            opt.step()

            bs = xb.shape[0]; count += bs
            tot["loss"] += float(loss.item()) * bs
            tot["rec"]  += float(loss_rec.item()) * bs
            tot["cyc"]  += float(loss_cyc.item()) * bs

        if sched is not None:
            sched.step()

        val_task = _eval_forward_val(model, val_loader, device)
        val_rjcp = estimate_rjcp_dataset(model, x_val,
                                          config.jcp_num_probes_eval,
                                          config.eval_subset_rjcp)
        score = val_rjcp + 1e-3 * val_task

        if score < best_score:
            best_score = score
            best_state = _state_copy(model)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            tag = "+JCP" if use_jcp else "-JCP"
            print(f"  [S3{tag}] epoch {epoch+1:03d} | "
                  f"train={tot['loss']/count:.5f} | val={val_task:.5f} | "
                  f"rjcp={val_rjcp:.5f} | jcp={tot['jcp']/count:.5f}")

    model.load_state_dict(best_state)
    return model
