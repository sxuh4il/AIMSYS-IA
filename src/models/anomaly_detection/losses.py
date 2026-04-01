"""Reconstruction loss and anomaly scoring utilities."""

from __future__ import annotations

from typing import Literal

from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def per_sample_reconstruction_error(inputs: Tensor, reconstructions: Tensor) -> Tensor:
    """
    Compute reconstruction error for each sample in a batch.

    Args:
        inputs: Original input tensor.
        reconstructions: Model reconstructions with the same shape as inputs.

    Returns:
        Tensor of shape [batch] containing per-sample MSE reconstruction errors.
    """
    if inputs.shape != reconstructions.shape:
        raise ValueError(
            "inputs and reconstructions must have the same shape, "
            f"got {tuple(inputs.shape)} vs {tuple(reconstructions.shape)}."
        )

    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
        reconstructions = reconstructions.unsqueeze(0)

    squared_error = (inputs - reconstructions).pow(2)
    reduction_dims = tuple(range(1, squared_error.dim()))
    return squared_error.mean(dim=reduction_dims)


def reconstruction_loss(
    inputs: Tensor,
    reconstructions: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """
    Compute reconstruction loss.

    Args:
        inputs: Original input tensor.
        reconstructions: Reconstructed tensor.
        reduction: Loss reduction mode.

    Returns:
        Reduced loss tensor or per-sample errors when reduction="none".
    """
    sample_errors = per_sample_reconstruction_error(inputs, reconstructions)

    if reduction == "none":
        return sample_errors
    if reduction == "mean":
        return sample_errors.mean()
    if reduction == "sum":
        return sample_errors.sum()
    raise ValueError(f"Unsupported reduction: {reduction}.")
