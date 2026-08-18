from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from information_field.causal_minimal import compile_minimal_realization

from information_field.observable_response.green import compile_fixed_time_green
from information_field.observable_response.recurrence import compile_grid_recurrence
from information_field.observable_response.spectral import compile_observable_spectrum


def make_repeated_spectrum(frequencies: int):
    if frequencies < 1:
        raise ValueError("frequency count must be positive")
    port_width = 4
    eigenvalues = torch.arange(1, frequencies + 1, dtype=torch.float64).repeat_interleave(port_width)
    operator = torch.diag(eigenvalues)
    identity = torch.eye(port_width, dtype=torch.float64)
    incident = identity.repeat(frequencies, 1)
    weights = torch.linspace(0.5, 1.5, frequencies, dtype=torch.float64)
    observation = torch.cat(
        tuple(weight * identity for weight in weights), dim=1
    )
    return compile_minimal_realization(operator, incident, observation)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _median_us(function, repetitions: int, device: str) -> float:
    for _ in range(10):
        function()
    _sync(device)
    values = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            function()
        _sync(device)
        values.append((time.perf_counter_ns() - start) / repetitions / 1_000.0)
    return statistics.median(values)


def _modal_batch(realization, incidents, time_value: float, mass: float):
    force = incidents @ realization.modal_incident_port.T
    values = realization.eigenvalues
    omega = torch.sqrt(values / mass)
    coefficient = (1.0 - torch.cos(omega * time_value)) / values
    return (force * coefficient[None, :]) @ realization.modal_observation.T


def _modal_single(realization, incident, time_value: float, mass: float):
    force = realization.modal_incident_port @ incident
    values = realization.eigenvalues
    omega = torch.sqrt(values / mass)
    coefficient = (1.0 - torch.cos(omega * time_value)) / values
    return realization.modal_observation @ (coefficient * force)


def benchmark_case(frequencies: int, *, device: str, repetitions: int) -> dict:
    realization = make_repeated_spectrum(frequencies)
    started = time.perf_counter_ns()
    spectrum = compile_observable_spectrum(realization)
    spectral_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    fixed = compile_fixed_time_green(
        realization, spectrum, time=0.73, mass=1.4
    )
    fixed_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    recurrence = compile_grid_recurrence(realization, step_size=0.05, mass=1.4)
    recurrence_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    dtype = torch.float32 if device in {"mps", "cuda"} else torch.float64
    lowered = realization.to(device, dtype)
    fixed = fixed.to(device, dtype)
    incident = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=dtype, device=device)
    batch = torch.linspace(-0.8, 0.9, 256 * 4, dtype=dtype, device=device).reshape(256, 4)
    expected = _modal_single(lowered, incident, 0.73, 1.4)
    actual = fixed.run_prepared(incident)
    response_error = float(torch.max(torch.abs(expected - actual)).item())

    modal_us = _median_us(
        lambda: _modal_single(lowered, incident, 0.73, 1.4),
        repetitions,
        device,
    )
    fixed_us = _median_us(
        lambda: fixed.run_prepared(incident), repetitions, device
    )
    basis_us = _median_us(lambda: fixed.run_basis(2), repetitions, device)
    modal_batch_us = _median_us(
        lambda: _modal_batch(lowered, batch, 0.73, 1.4), repetitions, device
    )
    fixed_batch_us = _median_us(lambda: fixed.run_batch(batch), repetitions, device)

    r = realization.state_dimension
    m = realization.incident_port.shape[1]
    p = realization.observation.shape[0]
    modal_multiply = r * (m + p + 8)
    fixed_multiply = m * p
    return {
        "device": device,
        "second_order_dimension": r,
        "first_order_online_degree": recurrence.certificate.first_order_degree,
        "frequencies": spectrum.certificate.distinct_frequencies,
        "residue_rank_sum": spectrum.certificate.residue_rank_sum,
        "maximum_moment_residual": spectrum.certificate.maximum_moment_residual,
        "response_absolute_error": response_error,
        "spectral_compile_ms": spectral_compile_ms,
        "fixed_map_compile_ms": fixed_compile_ms,
        "recurrence_compile_ms": recurrence_compile_ms,
        "modal_warm_us": modal_us,
        "fixed_map_warm_us": fixed_us,
        "fixed_map_speedup": modal_us / fixed_us,
        "basis_lookup_us": basis_us,
        "basis_lookup_speedup": modal_us / basis_us,
        "modal_batch_256_us": modal_batch_us,
        "fixed_map_batch_256_us": fixed_batch_us,
        "batch_speedup": modal_batch_us / fixed_batch_us,
        "modal_execution_bytes": lowered.execution_bytes,
        "fixed_map_bytes": fixed.retained_bytes,
        "byte_reduction": lowered.execution_bytes / max(1, fixed.retained_bytes),
        "modal_scalar_multiply_estimate": modal_multiply,
        "fixed_scalar_multiply_estimate": fixed_multiply,
        "multiply_reduction": modal_multiply / fixed_multiply,
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
    parser.add_argument("--frequencies", default="4,16,64")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    frequencies = tuple(int(value) for value in arguments.frequencies.split(","))
    results = [
        benchmark_case(count, device=device, repetitions=arguments.repetitions)
        for device in available_devices(arguments.device)
        for count in frequencies
    ]
    payload = {
        "contract": "zero-past fixed-time constant-source response",
        "batch_size": 256,
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
