from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Callable

import torch

from information_field.profiled_response import execute_prepared, prepare_backend_request
from information_field.response_backends import BackendKind, lower_backend_plan

from processor_kernels import _static_case, _temporal_cases
from information_field.native_response_kernels.cpu import CPUResponseKernels
from information_field.native_response_kernels.qualification import qualify_native_plan


@dataclass(frozen=True)
class CPUBenchmarkRecord:
    contract: str
    carrier_dimension: int
    input_dimension: int
    output_dimension: int
    history_steps: int
    iterations: int
    tensor_median_microseconds: float
    native_median_microseconds: float
    native_speedup: float
    native_kernel: str
    native_dispatches: int
    maximum_absolute_error: float
    precision_passed: bool


def _median_microseconds(
    operation: Callable[[], object], *, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def run_cpu_benchmark(
    *,
    dimensions: tuple[int, ...] = (32, 64, 128),
    batch_rows: int = 32,
    entries_per_row: int = 8,
    samples: int = 4,
    history_steps: int = 8,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    kernels = CPUResponseKernels()
    records = []
    for dimension in dimensions:
        cases = (
            _static_case(
                dimension,
                batch_rows=batch_rows,
                entries_per_row=entries_per_row,
            ),
        ) + _temporal_cases(
            dimension, samples=samples, history_steps=history_steps
        )
        for ir, request in cases:
            plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
            prepared = prepare_backend_request(plan, request)
            native = kernels.execute(plan, prepared)
            tensor_time = _median_microseconds(
                lambda: execute_prepared(plan, prepared),
                warmup=warmup,
                iterations=iterations,
            )
            native_time = _median_microseconds(
                lambda: kernels.execute(plan, prepared),
                warmup=warmup,
                iterations=iterations,
            )
            certificate = qualify_native_plan(
                ir,
                plan,
                kernels,
                (request,),
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
            steps = 0 if prepared.incident is None else prepared.incident.shape[0]
            records.append(
                CPUBenchmarkRecord(
                    ir.contract.value,
                    dimension,
                    ir.input_dimension,
                    ir.output_dimension,
                    steps,
                    iterations,
                    tensor_time,
                    native_time,
                    tensor_time / native_time,
                    native.record.kernel_name,
                    native.record.dispatches,
                    certificate.maximum_absolute_error,
                    certificate.passed,
                )
            )
    return {
        "contract": "native CPU response kernel benchmark 92.1",
        "backend": "cpu",
        "dtype": "torch.float64",
        "timing_scope": (
            "response evaluation only; input staging excluded; output allocation "
            "and execution included"
        ),
        "tensor_comparator": "CPU tensor plan for the identical response representation",
        "warmup": warmup,
        "iterations": iterations,
        "source_digest": kernels.source_digest,
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    print(
        json.dumps(
            run_cpu_benchmark(
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
