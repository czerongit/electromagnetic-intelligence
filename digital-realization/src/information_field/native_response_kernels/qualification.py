from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from information_field.profiled_response import prepare_backend_request
from information_field.response_backends import BackendKind, BackendPlan, BackendRequest
from information_field.response_ir import CompiledResponseIR, IRExecution, ResponseContract

from .metal import NativeKernelExecution


Tensor = torch.Tensor


class NativeResponseKernels(Protocol):
    source_digest: str

    def execute(self, plan, prepared) -> NativeKernelExecution: ...


@dataclass(frozen=True)
class NativePrecisionCertificate:
    artifact_digest: str
    plan_digest: str
    source_digest: str
    backend: BackendKind
    contract: ResponseContract
    dtype: torch.dtype
    cases: int
    values_compared: int
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool

    def is_valid_for(
        self,
        ir: CompiledResponseIR,
        plan: BackendPlan,
        kernels: NativeResponseKernels,
    ) -> bool:
        return (
            self.artifact_digest == ir.artifact_digest
            and self.plan_digest == plan.plan_digest
            and self.source_digest == kernels.source_digest
            and self.backend is plan.backend
            and self.contract is plan.contract
            and self.dtype is plan.dtype
        )


def _reference_execution(
    ir: CompiledResponseIR, request: BackendRequest
) -> IRExecution:
    return ir.execute(
        request.incident,
        local_indices=request.local_indices,
        valid=request.valid,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
        state_position=request.state_position,
        state_velocity=request.state_velocity,
    )


def _execution_tensors(execution: IRExecution) -> tuple[Tensor, ...]:
    return tuple(
        value
        for value in (
            execution.output,
            execution.final_position,
            execution.final_velocity,
        )
        if value is not None
    )


def qualify_native_plan(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    kernels: NativeResponseKernels,
    requests: tuple[BackendRequest, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> NativePrecisionCertificate:
    if not plan.is_valid_for(ir):
        raise ValueError("native plan was compiled from another response artifact")
    if not requests:
        raise ValueError("native precision qualification requires at least one case")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    values_compared = 0
    passed = True
    for request in requests:
        reference = _reference_execution(ir, request)
        prepared = prepare_backend_request(plan, request)
        actual = kernels.execute(plan, prepared).result
        actual_values = _execution_tensors(actual)
        reference_values = _execution_tensors(reference)
        if len(actual_values) != len(reference_values):
            raise ValueError("native execution returned the wrong response components")
        for actual_value, reference_value in zip(actual_values, reference_values):
            actual_cpu = actual_value.detach().cpu().to(torch.float64)
            reference_cpu = reference_value.detach().cpu().to(torch.float64)
            if actual_cpu.shape != reference_cpu.shape:
                raise ValueError("native execution returned the wrong response shape")
            difference = torch.abs(actual_cpu - reference_cpu)
            scale = torch.abs(reference_cpu)
            allowed = absolute_tolerance + relative_tolerance * scale
            if difference.numel():
                maximum_absolute = max(
                    maximum_absolute, float(torch.max(difference).item())
                )
                relative = difference / torch.clamp(scale, min=absolute_tolerance)
                maximum_relative = max(
                    maximum_relative, float(torch.max(relative).item())
                )
                passed = passed and bool(torch.all(difference <= allowed))
            values_compared += difference.numel()
    return NativePrecisionCertificate(
        ir.artifact_digest,
        plan.plan_digest,
        kernels.source_digest,
        plan.backend,
        plan.contract,
        plan.dtype,
        len(requests),
        values_compared,
        maximum_absolute,
        maximum_relative,
        absolute_tolerance,
        relative_tolerance,
        passed,
    )


def execute_qualified_native(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    kernels: NativeResponseKernels,
    certificate: NativePrecisionCertificate,
    request: BackendRequest,
) -> NativeKernelExecution:
    if not certificate.passed:
        raise ValueError("native precision certificate did not pass")
    if not certificate.is_valid_for(ir, plan, kernels):
        raise ValueError("native precision certificate is invalid for this execution")
    prepared = prepare_backend_request(plan, request)
    return kernels.execute(plan, prepared)
