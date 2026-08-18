from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Callable

import torch

from information_field.causal_minimal import compile_minimal_realization
from information_field.observable_response import (
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from information_field.profiled_response import execute_prepared, prepare_backend_request
from information_field.quotient_response import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)
from information_field.response_backends import BackendKind, BackendRequest, backend_capability, lower_backend_plan
from information_field.response_ir import (
    CompiledResponseIR,
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)

from information_field.native_response_kernels.metal import MPSResponseKernels
from information_field.native_response_kernels.qualification import qualify_native_plan


@dataclass(frozen=True)
class NativeBenchmarkRecord:
    contract: str
    carrier_dimension: int
    input_dimension: int
    output_dimension: int
    history_steps: int
    batch_rows: int
    entries_per_row: int
    iterations: int
    tensor_median_microseconds: float
    native_median_microseconds: float
    native_speedup: float
    serial_control_microseconds: float | None
    serial_control_speedup: float | None
    native_kernel: str
    native_dispatches: int
    serial_device_execution: bool
    maximum_absolute_error: float
    precision_passed: bool


def _synchronized_median_microseconds(
    operation: Callable[[], object], *, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
        torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        torch.mps.synchronize()
        started = time.perf_counter_ns()
        operation()
        torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def _static_case(
    dimension: int, *, batch_rows: int, entries_per_row: int
) -> tuple[CompiledResponseIR, BackendRequest]:
    generator = torch.Generator().manual_seed(9200 + dimension)
    operator = torch.eye(dimension, dtype=torch.float64)
    source = SparseRelationSource.from_dense(operator)
    compiled = compile_static_response(
        source,
        torch.eye(dimension, dtype=torch.float64),
        torch.arange(dimension, dtype=torch.int64),
    )
    indices = torch.randint(
        dimension,
        (batch_rows, entries_per_row),
        generator=generator,
        dtype=torch.int64,
    )
    amplitudes = torch.randn(
        (batch_rows, entries_per_row), generator=generator, dtype=torch.float64
    )
    valid = torch.ones((batch_rows, entries_per_row), dtype=torch.bool)
    if batch_rows:
        valid[batch_rows // 2] = False
    incidents = compiled.prepare(SparseIncidentBatch(indices, amplitudes, valid))
    return lower_static_response(compiled), BackendRequest(
        incidents.amplitudes,
        local_indices=incidents.local_indices,
        valid=incidents.valid,
    )


def _temporal_cases(
    dimension: int, *, samples: int, history_steps: int
) -> tuple[
    tuple[CompiledResponseIR, BackendRequest],
    tuple[CompiledResponseIR, BackendRequest],
    tuple[CompiledResponseIR, BackendRequest],
]:
    generator = torch.Generator().manual_seed(9210 + dimension)
    operator = torch.diag(torch.linspace(1.0, 2.0, dimension, dtype=torch.float64))
    identity = torch.eye(dimension, dtype=torch.float64)
    realization = compile_minimal_realization(operator, identity, identity)
    spectrum = compile_observable_spectrum(realization)
    coordinate = torch.randn(dimension, generator=generator, dtype=torch.float64)
    fixed = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.73, mass=1.4)
    )
    sampled = lower_sampled_times(
        compile_sampled_green(
            realization,
            spectrum,
            times=tuple((index + 1) * 0.17 for index in range(samples)),
            mass=1.3,
        )
    )
    grid = lower_grid_recurrence(
        compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
    )
    history = torch.randn(
        (history_steps, dimension), generator=generator, dtype=torch.float64
    )
    return (
        (fixed, BackendRequest(coordinate)),
        (sampled, BackendRequest(coordinate)),
        (grid, BackendRequest(history)),
    )


def _benchmark_case(
    kernels: MPSResponseKernels,
    serial_kernels: MPSResponseKernels,
    ir: CompiledResponseIR,
    request: BackendRequest,
    *,
    dimension: int,
    batch_rows: int,
    entries_per_row: int,
    warmup: int,
    iterations: int,
) -> NativeBenchmarkRecord:
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(plan, request)
    native = kernels.execute(plan, prepared)
    tensor_time = _synchronized_median_microseconds(
        lambda: execute_prepared(plan, prepared), warmup=warmup, iterations=iterations
    )
    native_time = _synchronized_median_microseconds(
        lambda: kernels.execute(plan, prepared), warmup=warmup, iterations=iterations
    )
    serial_time = None
    if ir.contract.value == "regular-grid":
        serial_time = _synchronized_median_microseconds(
            lambda: serial_kernels.execute(plan, prepared),
            warmup=warmup,
            iterations=iterations,
        )
    certificate = qualify_native_plan(
        ir,
        plan,
        kernels,
        (request,),
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
    )
    steps = 0 if prepared.incident is None else prepared.incident.shape[0]
    return NativeBenchmarkRecord(
        ir.contract.value,
        dimension,
        ir.input_dimension,
        ir.output_dimension,
        steps,
        batch_rows,
        entries_per_row,
        iterations,
        tensor_time,
        native_time,
        tensor_time / native_time,
        serial_time,
        None if serial_time is None else tensor_time / serial_time,
        native.record.kernel_name,
        native.record.dispatches,
        native.record.serial_device_execution,
        certificate.maximum_absolute_error,
        certificate.passed,
    )


def run_benchmark(
    *,
    dimensions: tuple[int, ...] = (32, 64, 128),
    batch_rows: int = 32,
    entries_per_row: int = 8,
    samples: int = 4,
    history_steps: int = 8,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    capability = backend_capability(BackendKind.MPS)
    if not capability.available:
        return {
            "contract": "native response kernel benchmark 92.1",
            "available": False,
            "reason": capability.reason,
            "records": [],
        }
    kernels = MPSResponseKernels()
    serial_kernels = MPSResponseKernels(recurrence_mode="serial")
    records = []
    for dimension in dimensions:
        static = _static_case(
            dimension,
            batch_rows=batch_rows,
            entries_per_row=entries_per_row,
        )
        cases = (static,) + _temporal_cases(
            dimension, samples=samples, history_steps=history_steps
        )
        for ir, request in cases:
            records.append(
                _benchmark_case(
                    kernels,
                    serial_kernels,
                    ir,
                    request,
                    dimension=dimension,
                    batch_rows=batch_rows if ir.contract.value == "static-columns" else 0,
                    entries_per_row=(
                        entries_per_row if ir.contract.value == "static-columns" else 0
                    ),
                    warmup=warmup,
                    iterations=iterations,
                )
            )
    return {
        "contract": "native response kernel benchmark 92.1",
        "available": True,
        "backend": "mps",
        "dtype": "torch.float32",
        "timing_scope": (
            "response evaluation only; input staging excluded; output allocation, "
            "kernel submission, execution, and synchronization included"
        ),
        "tensor_comparator": "MPS tensor plan for the identical response representation",
        "warmup": warmup,
        "iterations": iterations,
        "shader_source_digest": kernels.source_digest,
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    print(
        json.dumps(
            run_benchmark(
                dimensions=dimensions,
                warmup=arguments.warmup,
                iterations=arguments.iterations,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
