from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import statistics
import time

import torch

from information_field.profiled_response import PreparedBackendRequest, execute_prepared, prepare_backend_request
from information_field.response_backends import BackendKind, BackendPlan, BackendRequest
from information_field.response_ir import CompiledResponseIR, IRExecution

from .qualification import NativePrecisionCertificate, NativeResponseKernels, qualify_native_plan


class ExecutorKind(str, Enum):
    TENSOR = "tensor"
    NATIVE = "native"


@dataclass(frozen=True)
class ExecutorSelectionCertificate:
    artifact_digest: str
    plan_digest: str
    source_digest: str
    workload_signature: str
    backend: BackendKind
    tensor_median_microseconds: float
    native_median_microseconds: float
    selected: ExecutorKind
    warmup: int
    iterations: int
    native_precision: NativePrecisionCertificate

    def is_valid_for(
        self,
        ir: CompiledResponseIR,
        plan: BackendPlan,
        kernels: NativeResponseKernels,
        prepared: PreparedBackendRequest,
    ) -> bool:
        return (
            self.artifact_digest == ir.artifact_digest
            and self.plan_digest == plan.plan_digest
            and self.source_digest == kernels.source_digest
            and self.workload_signature == _workload_signature(plan, prepared)
            and self.backend is plan.backend
            and self.native_precision.is_valid_for(ir, plan, kernels)
            and self.native_precision.passed
        )


@dataclass(frozen=True)
class SelectedExecution:
    result: IRExecution
    executor: ExecutorKind


def _structural_tensor_signature(value: torch.Tensor | None) -> str:
    if value is None:
        return "none"
    return f"{tuple(value.shape)}:{value.dtype}:{value.device.type}"


def _workload_signature(
    plan: BackendPlan, prepared: PreparedBackendRequest
) -> str:
    digest = hashlib.sha256()
    digest.update(b"native-executor-workload-92.1")
    digest.update(plan.plan_digest.encode())
    for value in (
        prepared.incident,
        prepared.packed_coordinates,
        prepared.local_indices,
        prepared.local_weights,
        prepared.state_position,
        prepared.state_velocity,
    ):
        digest.update(_structural_tensor_signature(value).encode())
    if prepared.offsets is None:
        digest.update(b"offsets:none")
    else:
        offsets = prepared.offsets.detach().cpu().contiguous()
        digest.update(str(tuple(offsets.shape)).encode())
        digest.update(offsets.numpy().tobytes())
    return digest.hexdigest()


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


def calibrate_executor(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    kernels: NativeResponseKernels,
    request: BackendRequest,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    warmup: int = 5,
    iterations: int = 20,
) -> ExecutorSelectionCertificate:
    if warmup < 0 or iterations < 1:
        raise ValueError("executor calibration requires nonnegative warmup and iterations")
    prepared = prepare_backend_request(plan, request)
    precision = qualify_native_plan(
        ir,
        plan,
        kernels,
        (request,),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not precision.passed:
        raise ValueError("native executor failed response precision qualification")
    tensor_time = _median_microseconds(
        lambda: execute_prepared(plan, prepared), plan.backend, warmup, iterations
    )
    native_time = _median_microseconds(
        lambda: kernels.execute(plan, prepared), plan.backend, warmup, iterations
    )
    selected = (
        ExecutorKind.NATIVE
        if native_time < tensor_time
        else ExecutorKind.TENSOR
    )
    return ExecutorSelectionCertificate(
        ir.artifact_digest,
        plan.plan_digest,
        kernels.source_digest,
        _workload_signature(plan, prepared),
        plan.backend,
        tensor_time,
        native_time,
        selected,
        warmup,
        iterations,
        precision,
    )


def execute_selected(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    kernels: NativeResponseKernels,
    certificate: ExecutorSelectionCertificate,
    request: BackendRequest,
) -> SelectedExecution:
    prepared = prepare_backend_request(plan, request)
    if not certificate.is_valid_for(ir, plan, kernels, prepared):
        raise ValueError("executor selection is invalid for this response workload")
    if certificate.selected is ExecutorKind.NATIVE:
        result = kernels.execute(plan, prepared).result
    else:
        result = execute_prepared(plan, prepared).result
    return SelectedExecution(result, certificate.selected)
