from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch.profiler import ProfilerActivity, profile

from information_field.response_backends import BackendKind, BackendPlan, BackendRequest
from information_field.response_ir import IRExecution, ResponseContract


Tensor = torch.Tensor


@dataclass(frozen=True)
class PreparedBackendRequest:
    plan_digest: str
    contract: ResponseContract
    incident: Tensor | None
    packed_coordinates: Tensor | None
    local_indices: Tensor | None
    local_weights: Tensor | None
    offsets: Tensor | None
    state_position: Tensor | None
    state_velocity: Tensor | None
    input_transfer_bytes: int


@dataclass(frozen=True)
class PreparedExecution:
    result: IRExecution
    host_operator_calls: int
    lowering: str


@dataclass(frozen=True)
class DispatchTrace:
    backend: BackendKind
    contract: ResponseContract
    operation_counts: tuple[tuple[str, int], ...]
    core_operation_counts: tuple[tuple[str, int], ...]
    total_aten_calls: int
    core_calls: int
    synchronized_before: bool
    synchronized_after: bool


@dataclass(frozen=True)
class FusionCertificate:
    plan_digest: str
    backend: BackendKind
    contract: ResponseContract
    fused_semantic_operations: tuple[str, ...]
    measured_core_operation: str | None
    measured_core_calls: int
    backend_kernel_trace: bool
    fusion_certified: bool
    reason: str


def _bytes(values: tuple[Tensor | None, ...]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in values
        if value is not None
    )


def _coordinate(
    value: Tensor | None,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if value is None:
        return torch.zeros(width, dtype=dtype, device=device)
    if value.ndim != 1 or value.shape[0] != width:
        raise ValueError("coordinate has the wrong port dimension")
    return value.to(device=device, dtype=dtype)


def prepare_backend_request(
    plan: BackendPlan, request: BackendRequest
) -> PreparedBackendRequest:
    device, dtype = plan.device, plan.dtype
    if plan.contract is ResponseContract.STATIC_COLUMNS:
        if request.local_indices is None or request.valid is None:
            raise ValueError("static execution requires local indices and validity")
        if request.incident.shape != request.local_indices.shape or request.valid.shape != request.incident.shape:
            raise ValueError("static incident tensors must have one shape")
        valid = request.valid.to(torch.bool)
        counts = torch.sum(valid, dim=1, dtype=torch.int64)
        offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0))
        ).to(device)
        indices = request.local_indices[valid].to(device)
        weights = request.incident[valid].to(device=device, dtype=dtype)
        return PreparedBackendRequest(
            plan.plan_digest,
            plan.contract,
            None,
            None,
            indices,
            weights,
            offsets,
            None,
            None,
            _bytes((indices, weights, offsets)),
        )
    if plan.contract in {ResponseContract.FIXED_TIME, ResponseContract.SAMPLED_TIMES}:
        incident = request.incident.to(device=device, dtype=dtype)
        coordinates = torch.cat(
            (
                incident,
                _coordinate(
                    request.initial_position,
                    plan.initial_position_width,
                    device=device,
                    dtype=dtype,
                ),
                _coordinate(
                    request.initial_velocity,
                    plan.initial_velocity_width,
                    device=device,
                    dtype=dtype,
                ),
            )
        )
        return PreparedBackendRequest(
            plan.plan_digest,
            plan.contract,
            None,
            coordinates,
            None,
            None,
            None,
            None,
            None,
            _bytes((coordinates,)),
        )
    incident = request.incident.to(device=device, dtype=dtype)
    position = (
        None
        if request.state_position is None
        else request.state_position.to(device=device, dtype=dtype)
    )
    velocity = (
        None
        if request.state_velocity is None
        else request.state_velocity.to(device=device, dtype=dtype)
    )
    return PreparedBackendRequest(
        plan.plan_digest,
        plan.contract,
        incident,
        None,
        None,
        None,
        None,
        position,
        velocity,
        _bytes((incident, position, velocity)),
    )


def execute_prepared(
    plan: BackendPlan, prepared: PreparedBackendRequest
) -> PreparedExecution:
    if prepared.plan_digest != plan.plan_digest or prepared.contract is not plan.contract:
        raise ValueError("prepared request belongs to another backend plan")
    with torch.inference_mode():
        if plan.contract is ResponseContract.STATIC_COLUMNS:
            output = functional.embedding_bag(
                prepared.local_indices,
                plan.tensor("observed_columns"),
                prepared.offsets,
                per_sample_weights=prepared.local_weights,
                mode="sum",
                include_last_offset=True,
            )
            return PreparedExecution(
                IRExecution(output), 1, "weighted embedding-bag reduction"
            )
        if plan.contract in {ResponseContract.FIXED_TIME, ResponseContract.SAMPLED_TIMES}:
            output = plan.tensor("packed_map") @ prepared.packed_coordinates
            if plan.contract is ResponseContract.SAMPLED_TIMES:
                sample_count = plan.tensor("packed_map").shape[0] // plan.output_dimension
                output = output.reshape(sample_count, plan.output_dimension)
            return PreparedExecution(
                IRExecution(output), 1, "prepared packed matrix-vector response"
            )
        incidents = prepared.incident
        modes = plan.tensor("cosine").numel()
        position = (
            torch.zeros(modes, device=plan.device, dtype=plan.dtype)
            if prepared.state_position is None
            else prepared.state_position
        )
        velocity = (
            torch.zeros(modes, device=plan.device, dtype=plan.dtype)
            if prepared.state_velocity is None
            else prepared.state_velocity
        )
        outputs = []
        cosine = plan.tensor("cosine")
        for incident in incidents:
            force = plan.tensor("modal_incident") @ incident
            next_position = (
                cosine * position
                + plan.tensor("sine_over_omega") * velocity
                + plan.tensor("force_position") * force
            )
            next_velocity = (
                plan.tensor("negative_omega_sine") * position
                + cosine * velocity
                + plan.tensor("force_velocity") * force
            )
            position, velocity = next_position, next_velocity
            outputs.append(plan.tensor("modal_observation") @ position)
        output = (
            torch.stack(outputs)
            if outputs
            else torch.empty((0, plan.output_dimension), device=plan.device, dtype=plan.dtype)
        )
        return PreparedExecution(
            IRExecution(output, position, velocity),
            12 * incidents.shape[0] + (1 if incidents.shape[0] else 0),
            "structured modal recurrence",
        )


def _synchronize(backend: BackendKind) -> bool:
    if backend is BackendKind.CUDA:
        torch.cuda.synchronize()
        return True
    if backend is BackendKind.MPS:
        torch.mps.synchronize()
        return True
    return False


def trace_prepared_dispatch(
    plan: BackendPlan, prepared: PreparedBackendRequest
) -> DispatchTrace:
    execute_prepared(plan, prepared)
    before = _synchronize(plan.backend)
    with profile(activities=[ProfilerActivity.CPU]) as captured:
        execute_prepared(plan, prepared)
        after = _synchronize(plan.backend)
    counts = tuple(
        sorted(
            (event.key, event.count)
            for event in captured.key_averages()
            if event.key.startswith("aten::")
        )
    )
    if plan.backend is BackendKind.CPU:
        core_names = {
            ResponseContract.STATIC_COLUMNS: {
                "aten::cumsum",
                "aten::index_add_",
                "aten::copy_",
            },
            ResponseContract.FIXED_TIME: {"aten::addmv_"},
            ResponseContract.SAMPLED_TIMES: {"aten::addmv_"},
            ResponseContract.REGULAR_GRID: {"aten::addmv_", "aten::mul", "aten::add"},
        }[plan.contract]
    else:
        core_names = {
            ResponseContract.STATIC_COLUMNS: {"aten::_embedding_bag_forward_only"},
            ResponseContract.FIXED_TIME: {"aten::mm"},
            ResponseContract.SAMPLED_TIMES: {"aten::mm"},
            ResponseContract.REGULAR_GRID: {"aten::mm", "aten::mul", "aten::add"},
        }[plan.contract]
    core = tuple((name, count) for name, count in counts if name in core_names)
    return DispatchTrace(
        plan.backend,
        plan.contract,
        counts,
        core,
        sum(count for _, count in counts),
        sum(count for _, count in core),
        before,
        after,
    )


def certify_dispatch_fusion(
    plan: BackendPlan, trace: DispatchTrace
) -> FusionCertificate:
    if trace.backend is not plan.backend or trace.contract is not plan.contract:
        raise ValueError("dispatch trace belongs to another backend plan")
    expected = {
        ResponseContract.FIXED_TIME: (
            "aten::addmv_",
            ("incident response", "initial-position response", "initial-velocity response"),
        ),
        ResponseContract.SAMPLED_TIMES: (
            "aten::addmv_",
            ("sampled incident response", "sampled initial-position response", "sampled initial-velocity response"),
        ),
    }
    if plan.contract is ResponseContract.STATIC_COLUMNS:
        return FusionCertificate(
            plan.plan_digest,
            plan.backend,
            plan.contract,
            (),
            None,
            trace.core_calls,
            plan.backend is BackendKind.CPU,
            False,
            "weighted-column reduction still uses multiple CPU primitives or an unverified device wrapper",
        )
    if plan.contract is ResponseContract.REGULAR_GRID:
        return FusionCertificate(
            plan.plan_digest,
            plan.backend,
            plan.contract,
            (),
            None,
            trace.core_calls,
            plan.backend is BackendKind.CPU,
            False,
            "structured recurrence still dispatches force, state updates, and readout separately",
        )
    operation, semantics = expected[plan.contract]
    count = dict(trace.core_operation_counts).get(operation, 0)
    kernel_trace = plan.backend is BackendKind.CPU
    passed = kernel_trace and count == 1
    return FusionCertificate(
        plan.plan_digest,
        plan.backend,
        plan.contract,
        semantics,
        operation,
        count,
        kernel_trace,
        passed,
        "one measured CPU core operator implements the contracted response"
        if passed
        else "host operator trace is insufficient to certify backend kernel fusion",
    )
