from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class TemporalCondition:
    identifier: str
    batch: int
    modes: int
    run_lengths: tuple[int, ...]

    @property
    def steps(self) -> int:
        return sum(self.run_lengths)


TEMPORAL_CONDITIONS = (
    TemporalCondition("t256-e4", 64, 64, (64, 64, 64, 64)),
    TemporalCondition("t1024-e4", 64, 64, (256, 256, 256, 256)),
    TemporalCondition("t4096-e4", 64, 64, (1024, 1024, 1024, 1024)),
)


def make_temporal_fixture(
    condition: TemporalCondition,
    *,
    seed: int = 9702,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create source events, their durations, an initial state, and a readout.

    Each event supplies one constant affine modal transition.  Homogeneous
    coordinates make the forced recurrence a single three-dimensional linear
    action per mode.
    """

    generator = torch.Generator().manual_seed(seed)
    events = len(condition.run_lengths)
    frequencies = torch.linspace(0.35, 2.15, condition.modes, dtype=dtype)
    step = 0.0025
    cosine = torch.cos(frequencies * step)
    sine = torch.sin(frequencies * step)
    base = torch.zeros((condition.modes, 3, 3), dtype=dtype)
    base[:, 0, 0] = cosine
    base[:, 0, 1] = sine / frequencies
    base[:, 1, 0] = -frequencies * sine
    base[:, 1, 1] = cosine
    base[:, 2, 2] = 1.0
    force = torch.stack(
        ((1.0 - cosine) / frequencies.square(), sine / frequencies), dim=1
    )
    amplitudes = 0.05 * torch.randn(
        (events, condition.modes), generator=generator, dtype=dtype
    )
    transitions = base.unsqueeze(0).repeat(events, 1, 1, 1)
    transitions[:, :, :2, 2] = amplitudes[:, :, None] * force.unsqueeze(0)
    lengths = torch.tensor(condition.run_lengths, dtype=torch.int64)
    initial = 0.1 * torch.randn(
        (condition.batch, condition.modes, 3), generator=generator, dtype=dtype
    )
    initial[:, :, 2] = 1.0
    readout = torch.randn(
        (16, condition.modes), generator=generator, dtype=dtype
    ) / math.sqrt(condition.modes)
    return transitions.contiguous(), lengths, initial.contiguous(), readout


def matrix_powers(transitions: Tensor, lengths: Tensor) -> Tensor:
    if transitions.shape[0] != lengths.numel():
        raise ValueError("each source event requires one duration")
    return torch.stack(
        [torch.linalg.matrix_power(matrix, int(length)) for matrix, length in zip(transitions, lengths)]
    ).contiguous()


def regular_grid_response(transitions: Tensor, lengths: Tensor, initial: Tensor) -> Tensor:
    state = initial.clone()
    for transition, length in zip(transitions, lengths):
        for _ in range(int(length)):
            state = torch.einsum("mij,bmj->bmi", transition, state)
    return state


def event_composed_response(powers: Tensor, initial: Tensor) -> Tensor:
    state = initial.clone()
    for power in powers:
        state = torch.einsum("mij,bmj->bmi", power, state)
    return state


def observe(state: Tensor, readout: Tensor) -> Tensor:
    return torch.einsum("zm,bm->bz", readout, state[:, :, 0])


def correctness_record(condition: TemporalCondition) -> dict:
    transitions, lengths, initial, readout = make_temporal_fixture(condition)
    started = time.perf_counter_ns()
    powers = matrix_powers(transitions, lengths)
    preparation_ms = (time.perf_counter_ns() - started) / 1e6
    regular = regular_grid_response(transitions, lengths, initial)
    event = event_composed_response(powers, initial)
    return {
        "condition": asdict(condition),
        "steps": condition.steps,
        "events": len(condition.run_lengths),
        "power_preparation_ms": preparation_ms,
        "state_max_abs": float(torch.max(torch.abs(event - regular)).item()),
        "observation_max_abs": float(
            torch.max(torch.abs(observe(event, readout) - observe(regular, readout))).item()
        ),
        "regular_state_applications": condition.steps * condition.batch * condition.modes,
        "event_state_applications": len(condition.run_lengths) * condition.batch * condition.modes,
        "state_application_reduction": condition.steps / len(condition.run_lengths),
    }
