from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time

import torch

from information_field.native_response_kernels import qualify_native_plan
from information_field.profiled_response import execute_prepared, prepare_backend_request
from information_field.response_backends import BackendKind, BackendRequest, backend_capability, lower_backend_plan

from information_field.wide_response_kernels.cuda import WideCUDAResponseKernels
from information_field.wide_response_kernels.fixture import make_wide_grid_ir
from information_field.wide_response_kernels.metal import WideMPSResponseKernels


@dataclass(frozen=True)
class WideBenchmarkRecord:
    backend: str
    modes: int
    input_dimension: int
    output_dimension: int
    history_steps: int
    retained_bytes: int
    iterations: int
    tensor_host_operator_calls: int
    native_device_dispatches: int
    tensor_median_microseconds: float
    native_median_microseconds: float
    native_speedup: float
    selected_executor: str
    maximum_absolute_error: float
    precision_passed: bool


def _synchronize(backend: BackendKind) -> None:
    if backend is BackendKind.CUDA:
        torch.cuda.synchronize()
    elif backend is BackendKind.MPS:
        torch.mps.synchronize()


def _median_microseconds(operation, backend, warmup, iterations) -> float:
    for _ in range(warmup):
        operation()
        _synchronize(backend)
    samples = []
    for _ in range(iterations):
        _synchronize(backend)
        started = time.perf_counter_ns()
        operation()
        _synchronize(backend)
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def run_benchmark(
    backend: BackendKind,
    *,
    dimensions: tuple[int, ...] = (1025, 2048, 4096),
    input_dimension: int = 8,
    output_dimension: int = 16,
    history_steps: int = 8,
    warmup: int = 5,
    iterations: int = 30,
) -> dict:
    capability = backend_capability(backend)
    if not capability.available:
        return {
            "contract": "wide recurrent response benchmark 93.1",
            "available": False,
            "backend": backend.value,
            "reason": capability.reason,
            "records": [],
        }
    if backend is BackendKind.MPS:
        kernels = WideMPSResponseKernels()
    elif backend is BackendKind.CUDA:
        kernels = WideCUDAResponseKernels()
    else:
        raise ValueError("wide device benchmark requires MPS or CUDA")
    records = []
    for modes in dimensions:
        ir = make_wide_grid_ir(
            modes=modes,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            seed=9300 + modes,
        )
        generator = torch.Generator().manual_seed(9400 + modes)
        request = BackendRequest(
            torch.randn(
                (history_steps, input_dimension),
                generator=generator,
                dtype=torch.float64,
            ),
            state_position=torch.randn(
                modes, generator=generator, dtype=torch.float64
            ) / modes**0.5,
            state_velocity=torch.randn(
                modes, generator=generator, dtype=torch.float64
            ) / modes**0.5,
        )
        plan = lower_backend_plan(ir, backend, torch.float32)
        prepared = prepare_backend_request(plan, request)
        tensor = execute_prepared(plan, prepared)
        native = kernels.execute(plan, prepared)
        tensor_time = _median_microseconds(
            lambda: execute_prepared(plan, prepared), backend, warmup, iterations
        )
        native_time = _median_microseconds(
            lambda: kernels.execute(plan, prepared), backend, warmup, iterations
        )
        certificate = qualify_native_plan(
            ir,
            plan,
            kernels,
            (request,),
            absolute_tolerance=2e-5,
            relative_tolerance=2e-4,
        )
        records.append(
            WideBenchmarkRecord(
                backend.value,
                modes,
                input_dimension,
                output_dimension,
                history_steps,
                plan.retained_bytes,
                iterations,
                tensor.host_operator_calls,
                native.record.dispatches,
                tensor_time,
                native_time,
                tensor_time / native_time,
                "native" if native_time < tensor_time else "tensor",
                certificate.maximum_absolute_error,
                certificate.passed,
            )
        )
    payload = {
        "contract": "wide recurrent response benchmark 93.1",
        "available": True,
        "backend": backend.value,
        "dtype": "torch.float32",
        "timing_scope": (
            "regular-grid response from staged inputs; output allocation, submission, "
            "execution, and synchronization included"
        ),
        "tensor_comparator": "tensor realization of the same modal recurrence",
        "warmup": warmup,
        "iterations": iterations,
        "source_digest": kernels.source_digest,
        "records": [asdict(record) for record in records],
    }
    if backend is BackendKind.CUDA:
        payload["device"] = torch.cuda.get_device_name(0)
        payload["torch"] = torch.__version__
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mps", "cuda"), required=True)
    parser.add_argument("--dimensions", default="1025,2048,4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    print(
        json.dumps(
            run_benchmark(
                BackendKind(arguments.backend),
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
