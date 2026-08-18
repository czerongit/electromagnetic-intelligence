from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Callable

import torch

from information_field.profiled_response import execute_prepared, prepare_backend_request
from information_field.response_backends import BackendKind, backend_capability, lower_backend_plan

from processor_kernels import _static_case, _temporal_cases
from information_field.native_response_kernels.cuda import CUDAResponseKernels
from information_field.native_response_kernels.qualification import qualify_native_plan


@dataclass(frozen=True)
class CUDABenchmarkRecord:
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
    serial_device_execution: bool
    maximum_absolute_error: float
    precision_passed: bool


def _synchronized_median_microseconds(
    operation: Callable[[], object], *, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def run_cuda_benchmark(
    *,
    dimensions: tuple[int, ...] = (32, 64, 128),
    batch_rows: int = 32,
    entries_per_row: int = 8,
    samples: int = 4,
    history_steps: int = 8,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    capability = backend_capability(BackendKind.CUDA)
    if not capability.available:
        return {
            "contract": "native CUDA response kernel benchmark 92.1",
            "available": False,
            "reason": capability.reason,
            "records": [],
        }
    kernels = CUDAResponseKernels()
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
            plan = lower_backend_plan(ir, BackendKind.CUDA, torch.float32)
            prepared = prepare_backend_request(plan, request)
            native = kernels.execute(plan, prepared)
            tensor_time = _synchronized_median_microseconds(
                lambda: execute_prepared(plan, prepared),
                warmup=warmup,
                iterations=iterations,
            )
            native_time = _synchronized_median_microseconds(
                lambda: kernels.execute(plan, prepared),
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
            records.append(
                CUDABenchmarkRecord(
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
                    native.record.serial_device_execution,
                    certificate.maximum_absolute_error,
                    certificate.passed,
                )
            )
    return {
        "contract": "native CUDA response kernel benchmark 92.1",
        "available": True,
        "backend": "cuda",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "torch.float32",
        "timing_scope": (
            "response evaluation only; input staging excluded; output allocation, "
            "kernel submission, execution, and synchronization included"
        ),
        "tensor_comparator": "CUDA tensor plan for the identical response representation",
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
            run_cuda_benchmark(
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
