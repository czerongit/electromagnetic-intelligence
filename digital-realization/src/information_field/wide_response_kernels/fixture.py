from __future__ import annotations

import torch

from information_field.observable_response import ExactGridRecurrence, OnlineMinimalityCertificate
from information_field.response_ir import CompiledResponseIR, lower_grid_recurrence


def make_wide_grid_ir(
    *,
    modes: int = 1025,
    input_dimension: int = 3,
    output_dimension: int = 5,
    seed: int = 9301,
) -> CompiledResponseIR:
    if modes < 1 or input_dimension < 1 or output_dimension < 1:
        raise ValueError("wide recurrence dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    eigenvalues = torch.linspace(0.5, 2.0, modes, dtype=torch.float64)
    step = 0.03
    omega = torch.sqrt(eigenvalues)
    cosine = torch.cos(omega * step)
    sine_over_omega = torch.sin(omega * step) / omega
    negative_omega_sine = -omega * torch.sin(omega * step)
    force_position = (1.0 - cosine) / eigenvalues
    force_velocity = torch.sin(omega * step) / omega
    modal_incident = torch.randn(
        (modes, input_dimension), generator=generator, dtype=torch.float64
    ) / modes**0.5
    modal_observation = torch.randn(
        (output_dimension, modes), generator=generator, dtype=torch.float64
    ) / modes**0.5
    recurrence = ExactGridRecurrence(
        step,
        1.0,
        cosine,
        sine_over_omega,
        negative_omega_sine,
        force_position,
        force_velocity,
        modal_incident,
        modal_observation,
        torch.empty((modes, 0), dtype=torch.float64),
        torch.empty((modes, 0), dtype=torch.float64),
        OnlineMinimalityCertificate(
            2 * modes,
            2 * modes,
            2 * modes,
            2 * modes,
            True,
            False,
            f"wide-realization-{modes}-{input_dimension}-{output_dimension}-{seed}",
        ),
    )
    return lower_grid_recurrence(recurrence)
