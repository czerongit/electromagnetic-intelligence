from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import statistics
import time

import torch

from information_field.causal_minimal.realization import CausalMinimalRealization, compile_minimal_realization


Tensor = torch.Tensor


@dataclass(frozen=True)
class Fixture:
    operator: Tensor
    incident: Tensor
    observation: Tensor
    widened_incident: Tensor
    widened_observation: Tensor
    visible_dimension: int
    incident_control_dimension: int
    observation_control_dimension: int


@dataclass(frozen=True)
class PreparedFullResponse:
    eigenvalues: Tensor
    modal_incident: Tensor
    modal_observation: Tensor
    tolerance: float

    @property
    def execution_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.eigenvalues,
                self.modal_incident,
                self.modal_observation,
            )
        )

    def to(self, device: str, dtype: torch.dtype) -> "PreparedFullResponse":
        return PreparedFullResponse(
            self.eigenvalues.to(device=device, dtype=dtype),
            self.modal_incident.to(device=device, dtype=dtype),
            self.modal_observation.to(device=device, dtype=dtype),
            self.tolerance,
        )

    def respond(self, incident: Tensor, *, time_value: float, mass: float) -> Tensor:
        force = self.modal_incident @ incident
        active = self.eigenvalues > self.tolerance
        displacement = 0.5 * time_value * time_value * force / mass
        if bool(active.any()):
            stiffness = self.eigenvalues[active]
            omega = torch.sqrt(stiffness / mass)
            displacement[active] = (
                (1.0 - torch.cos(omega * time_value))
                * force[active]
                / stiffness
            )
        return self.modal_observation @ displacement


def make_fixture(dimension: int, dtype: torch.dtype = torch.float64) -> Fixture:
    if dimension < 32:
        raise ValueError("benchmark dimension must be at least 32")
    active, incident_control, observation_control = 8, 6, 5
    diagonal = torch.linspace(0.5, 3.0, dimension, dtype=dtype)
    operator = torch.diag(diagonal)

    # Base incidents reach A and C. Base observations see A and B. Only A is
    # both reachable and observable. Each widened port exposes one additional
    # sector already covered by its opposite port.
    a = torch.arange(0, active)
    b = torch.arange(active, active + incident_control)
    c = torch.arange(active + incident_control, active + incident_control + observation_control)
    incident_indices = torch.cat((a, c))
    observation_indices = torch.cat((a, b))
    incident = torch.eye(dimension, dtype=dtype)[:, incident_indices]
    observation = torch.eye(dimension, dtype=dtype)[observation_indices]
    widened_incident = torch.eye(dimension, dtype=dtype)[:, torch.cat((incident_indices, b))]
    widened_observation = torch.eye(dimension, dtype=dtype)[torch.cat((observation_indices, c))]
    return Fixture(
        operator,
        incident,
        observation,
        widened_incident,
        widened_observation,
        active,
        incident_control,
        observation_control,
    )


def prepare_full(fixture: Fixture) -> PreparedFullResponse:
    # Fixture operator is diagonal. Retaining all diagonal modes gives the
    # exact unreduced execution and avoids charging it an irrelevant eigensolve.
    eigenvalues = torch.diagonal(fixture.operator).clone()
    return PreparedFullResponse(
        eigenvalues,
        fixture.incident.clone(),
        fixture.observation.clone(),
        1e-12,
    )


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _median_microseconds(function, *, repetitions: int, device: str) -> float:
    for _ in range(10):
        function()
    _synchronize(device)
    samples = []
    groups = 7
    for _ in range(groups):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            function()
        _synchronize(device)
        samples.append((time.perf_counter_ns() - start) / repetitions / 1_000.0)
    return statistics.median(samples)


def _device_dtype(device: str) -> torch.dtype:
    return torch.float32 if device in {"mps", "cuda"} else torch.float64


def benchmark_case(
    dimension: int,
    *,
    device: str,
    repetitions: int,
) -> dict[str, float | int | str]:
    fixture = make_fixture(dimension)
    started = time.perf_counter_ns()
    minimal_cpu = compile_minimal_realization(
        fixture.operator, fixture.incident, fixture.observation
    )
    minimal_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    full_cpu = prepare_full(fixture)
    full_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    widened_incident = compile_minimal_realization(
        fixture.operator, fixture.widened_incident, fixture.observation
    )
    widened_observation = compile_minimal_realization(
        fixture.operator, fixture.incident, fixture.widened_observation
    )

    dtype = _device_dtype(device)
    minimal = minimal_cpu.to(device, dtype)
    full = full_cpu.to(device, dtype)
    incident = torch.linspace(
        -0.7,
        0.9,
        fixture.incident.shape[1],
        dtype=dtype,
        device=device,
    )
    time_value = 0.73
    mass = 1.4
    expected = full.respond(incident, time_value=time_value, mass=mass)
    actual = minimal.respond_prepared_zero_past_constant(
        incident, time=time_value, mass=mass
    )
    absolute_error = float(torch.max(torch.abs(expected - actual)).item())
    relative_error = absolute_error / max(1.0, float(torch.max(torch.abs(expected)).item()))

    full_us = _median_microseconds(
        lambda: full.respond(incident, time_value=time_value, mass=mass),
        repetitions=repetitions,
        device=device,
    )
    minimal_us = _median_microseconds(
        lambda: minimal.respond_prepared_zero_past_constant(
            incident, time=time_value, mass=mass
        ),
        repetitions=repetitions,
        device=device,
    )
    n = dimension
    r = minimal.state_dimension
    input_width = fixture.incident.shape[1]
    output_width = fixture.observation.shape[0]
    full_multiply_estimate = n * (input_width + output_width + 8)
    minimal_multiply_estimate = r * (input_width + output_width + 8)
    return {
        "device": device,
        "ambient_dimension": n,
        "reachable_dimension": minimal.certificate.reachable_dimension,
        "minimal_dimension": r,
        "retained_fraction": minimal.certificate.retained_fraction,
        "incident_width": input_width,
        "observation_width": output_width,
        "widened_incident_minimal_dimension": widened_incident.state_dimension,
        "widened_observation_minimal_dimension": widened_observation.state_dimension,
        "maximum_markov_residual": minimal.certificate.maximum_markov_residual,
        "response_relative_error": relative_error,
        "minimal_compile_ms": minimal_compile_ms,
        "full_prepare_ms": full_compile_ms,
        "full_warm_us": full_us,
        "minimal_warm_us": minimal_us,
        "warm_speedup": full_us / minimal_us,
        "full_execution_bytes": full.execution_bytes,
        "minimal_execution_bytes": minimal.execution_bytes,
        "execution_byte_reduction": full.execution_bytes / max(1, minimal.execution_bytes),
        "full_scalar_multiply_estimate": full_multiply_estimate,
        "minimal_scalar_multiply_estimate": minimal_multiply_estimate,
        "multiply_reduction": full_multiply_estimate / max(1, minimal_multiply_estimate),
    }


def available_devices(requested: str) -> tuple[str, ...]:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return (requested,)
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return tuple(devices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="128,256,512")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    results = [
        benchmark_case(
            dimension,
            device=device,
            repetitions=arguments.repetitions,
        )
        for device in available_devices(arguments.device)
        for dimension in dimensions
    ]
    payload = {
        "contract": "zero-past constant-source causal response",
        "repetitions": arguments.repetitions,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
