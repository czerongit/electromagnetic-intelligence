from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

import torch

from information_field.response_ir import CompiledResponseIR, IRExecution, ResponseContract


Tensor = torch.Tensor


class BackendKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(frozen=True)
class BackendCapability:
    backend: BackendKind
    available: bool
    reason: str
    supported_dtypes: tuple[torch.dtype, ...]

    def supports(self, dtype: torch.dtype) -> bool:
        return self.available and dtype in self.supported_dtypes


def backend_capability(backend: BackendKind) -> BackendCapability:
    if backend is BackendKind.CPU:
        return BackendCapability(
            backend,
            True,
            "CPU tensor execution is available",
            (torch.float64, torch.float32),
        )
    if backend is BackendKind.CUDA:
        available = torch.cuda.is_available()
        return BackendCapability(
            backend,
            available,
            "CUDA device is available" if available else "CUDA device is unavailable",
            (torch.float64, torch.float32, torch.float16),
        )
    mps = getattr(torch.backends, "mps", None)
    available = bool(mps is not None and mps.is_available())
    return BackendCapability(
        backend,
        available,
        "MPS device is available" if available else "MPS device is unavailable",
        (torch.float32, torch.float16),
    )


@dataclass(frozen=True)
class PlanDecision:
    requested_backend: BackendKind
    selected_backend: BackendKind
    requested_dtype: torch.dtype
    selected_dtype: torch.dtype
    fallback_used: bool
    reason: str


def choose_backend(
    requested_backend: BackendKind,
    requested_dtype: torch.dtype,
    *,
    allow_fallback: bool = True,
) -> PlanDecision:
    capability = backend_capability(requested_backend)
    if capability.supports(requested_dtype):
        return PlanDecision(
            requested_backend,
            requested_backend,
            requested_dtype,
            requested_dtype,
            False,
            "requested backend and dtype are available",
        )
    if not allow_fallback:
        raise RuntimeError(
            f"{requested_backend.value} does not support requested execution: "
            f"{capability.reason}, dtype={requested_dtype}"
        )
    cpu = backend_capability(BackendKind.CPU)
    selected_dtype = (
        requested_dtype if cpu.supports(requested_dtype) else torch.float32
    )
    return PlanDecision(
        requested_backend,
        BackendKind.CPU,
        requested_dtype,
        selected_dtype,
        True,
        f"fallback to CPU because {capability.reason} or dtype is unsupported",
    )


@dataclass(frozen=True)
class BackendRequest:
    incident: Tensor
    local_indices: Tensor | None = None
    valid: Tensor | None = None
    initial_position: Tensor | None = None
    initial_velocity: Tensor | None = None
    state_position: Tensor | None = None
    state_velocity: Tensor | None = None


@dataclass(frozen=True)
class BackendPlan:
    artifact_digest: str
    plan_digest: str
    contract: ResponseContract
    backend: BackendKind
    dtype: torch.dtype
    tensors: tuple[tuple[str, Tensor], ...]
    input_dimension: int
    output_dimension: int
    state_dimension: int
    incident_width: int
    initial_position_width: int
    initial_velocity_width: int
    retained_bytes: int
    lowering: str
    fusion_claimed: bool

    @property
    def device(self) -> torch.device:
        return torch.device(self.backend.value)

    def tensor(self, name: str) -> Tensor:
        for tensor_name, value in self.tensors:
            if tensor_name == name:
                return value
        raise KeyError(name)

    def is_valid_for(self, ir: CompiledResponseIR) -> bool:
        return self.artifact_digest == ir.artifact_digest


@dataclass(frozen=True)
class BackendExecution:
    result: IRExecution
    backend: BackendKind
    dtype: torch.dtype
    lowering: str
    estimated_tensor_dispatches: int


@dataclass(frozen=True)
class BackendPrecisionCertificate:
    artifact_digest: str
    plan_digest: str
    backend: BackendKind
    dtype: torch.dtype
    cases: int
    values_compared: int
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool

    def is_valid_for(self, ir: CompiledResponseIR, plan: BackendPlan) -> bool:
        return (
            self.artifact_digest == ir.artifact_digest
            and self.plan_digest == plan.plan_digest
            and self.backend is plan.backend
            and self.dtype is plan.dtype
        )


def _plan_digest(ir: CompiledResponseIR, backend: BackendKind, dtype: torch.dtype) -> str:
    digest = hashlib.sha256()
    digest.update(b"response-backend-plan-v1")
    digest.update(ir.artifact_digest.encode())
    digest.update(backend.value.encode())
    digest.update(str(dtype).encode())
    return digest.hexdigest()


def _retained_bytes(tensors: tuple[tuple[str, Tensor], ...]) -> int:
    return sum(value.numel() * value.element_size() for _, value in tensors)


def _binding_width(ir: CompiledResponseIR, name: str) -> int:
    try:
        return ir.tensor(name).shape[-1]
    except KeyError:
        return 0


def lower_backend_plan(
    ir: CompiledResponseIR,
    backend: BackendKind,
    dtype: torch.dtype,
) -> BackendPlan:
    ir.assert_integrity()
    capability = backend_capability(backend)
    if not capability.supports(dtype):
        raise RuntimeError(
            f"{backend.value} cannot execute dtype {dtype}: {capability.reason}"
        )
    device = torch.device(backend.value)
    tensors: tuple[tuple[str, Tensor], ...]
    lowering: str
    if ir.contract is ResponseContract.STATIC_COLUMNS:
        tensors = (("observed_columns", ir.tensor("observed_columns").to(device, dtype)),)
        lowering = "structured weighted-column reduction"
    elif ir.contract is ResponseContract.FIXED_TIME:
        packed = torch.cat(
            (
                ir.tensor("incident_map"),
                ir.tensor("initial_position_map"),
                ir.tensor("initial_velocity_map"),
            ),
            dim=1,
        ).to(device, dtype)
        tensors = (("packed_map", packed),)
        lowering = "single packed linear application"
    elif ir.contract is ResponseContract.SAMPLED_TIMES:
        packed = torch.cat(
            (
                ir.tensor("incident_maps"),
                ir.tensor("initial_position_maps"),
                ir.tensor("initial_velocity_maps"),
            ),
            dim=2,
        )
        tensors = (
            ("packed_map", packed.reshape(-1, packed.shape[-1]).to(device, dtype)),
            ("sample_count", torch.tensor([packed.shape[0]], device=device)),
        )
        lowering = "single flattened sampled linear application"
    elif ir.contract is ResponseContract.REGULAR_GRID:
        names = (
            "cosine",
            "sine_over_omega",
            "negative_omega_sine",
            "force_position",
            "force_velocity",
            "modal_incident",
            "modal_observation",
        )
        tensors = tuple((name, ir.tensor(name).to(device, dtype)) for name in names)
        lowering = "structured diagonal modal recurrence"
    else:
        raise AssertionError("unhandled response contract")
    return BackendPlan(
        ir.artifact_digest,
        _plan_digest(ir, backend, dtype),
        ir.contract,
        backend,
        dtype,
        tensors,
        ir.input_dimension,
        ir.output_dimension,
        ir.state_dimension,
        _binding_width(ir, "incident_map") or _binding_width(ir, "incident_maps") or (
            ir.tensor("modal_incident").shape[1]
            if ir.contract is ResponseContract.REGULAR_GRID
            else 0
        ),
        _binding_width(ir, "initial_position_map")
        or _binding_width(ir, "initial_position_maps"),
        _binding_width(ir, "initial_velocity_map")
        or _binding_width(ir, "initial_velocity_maps"),
        _retained_bytes(tensors),
        lowering,
        False,
    )


def _coordinate(
    value: Tensor | None,
    width: int,
    reference: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if value is None:
        return torch.zeros(width, device=device, dtype=dtype)
    if value.ndim != 1 or value.shape[0] != width:
        raise ValueError("coordinate has the wrong port dimension")
    return value.to(device=device, dtype=dtype)


def execute_backend_plan(plan: BackendPlan, request: BackendRequest) -> BackendExecution:
    device, dtype = plan.device, plan.dtype
    incident = request.incident.to(device=device, dtype=dtype)
    with torch.inference_mode():
        if plan.contract is ResponseContract.STATIC_COLUMNS:
            if request.local_indices is None or request.valid is None:
                raise ValueError("static execution requires local indices and validity")
            indices = request.local_indices.to(device=device)
            valid = request.valid.to(device=device)
            if indices.shape != incident.shape or valid.shape != incident.shape:
                raise ValueError("static incident tensors must have one shape")
            safe = torch.where(valid, indices, torch.zeros_like(indices))
            weights = torch.where(valid, incident, torch.zeros_like(incident))
            output = torch.sum(
                plan.tensor("observed_columns")[safe] * weights[..., None], dim=1
            )
            return BackendExecution(
                IRExecution(output), plan.backend, dtype, plan.lowering, 4
            )
        if plan.contract in {ResponseContract.FIXED_TIME, ResponseContract.SAMPLED_TIMES}:
            coordinates = torch.cat(
                (
                    incident,
                    _coordinate(
                        request.initial_position,
                        plan.initial_position_width,
                        incident,
                        device,
                        dtype,
                    ),
                    _coordinate(
                        request.initial_velocity,
                        plan.initial_velocity_width,
                        incident,
                        device,
                        dtype,
                    ),
                )
            )
            output = plan.tensor("packed_map") @ coordinates
            if plan.contract is ResponseContract.SAMPLED_TIMES:
                sample_count = int(plan.tensor("sample_count")[0].item())
                output = output.reshape(sample_count, plan.output_dimension)
            return BackendExecution(
                IRExecution(output), plan.backend, dtype, plan.lowering, 1
            )
        if incident.ndim != 2 or incident.shape[1] != plan.incident_width:
            raise ValueError("incident history has the wrong port dimension")
        modes = plan.tensor("cosine").numel()
        position = (
            torch.zeros(modes, device=device, dtype=dtype)
            if request.state_position is None
            else request.state_position.to(device=device, dtype=dtype)
        )
        velocity = (
            torch.zeros(modes, device=device, dtype=dtype)
            if request.state_velocity is None
            else request.state_velocity.to(device=device, dtype=dtype)
        )
        outputs = []
        cosine = plan.tensor("cosine")
        for current in incident:
            force = plan.tensor("modal_incident") @ current
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
            else torch.empty((0, plan.output_dimension), device=device, dtype=dtype)
        )
        return BackendExecution(
            IRExecution(output, position, velocity),
            plan.backend,
            dtype,
            plan.lowering,
            9 * incident.shape[0],
        )


def _reference_execution(ir: CompiledResponseIR, request: BackendRequest) -> IRExecution:
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


def qualify_backend_plan(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    requests: tuple[BackendRequest, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> BackendPrecisionCertificate:
    if not plan.is_valid_for(ir):
        raise ValueError("backend plan was compiled from another response artifact")
    if not requests:
        raise ValueError("precision qualification requires at least one case")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    values_compared = 0
    for request in requests:
        expected = _reference_execution(ir, request)
        actual = execute_backend_plan(plan, request).result
        expected_values = _execution_tensors(expected)
        actual_values = _execution_tensors(actual)
        if len(expected_values) != len(actual_values):
            raise ValueError("backend result omitted a declared state output")
        for expected_tensor, actual_tensor in zip(expected_values, actual_values):
            expected_cpu = expected_tensor.detach().cpu().to(torch.float64)
            actual_cpu = actual_tensor.detach().cpu().to(torch.float64)
            if expected_cpu.shape != actual_cpu.shape:
                raise ValueError("backend result has the wrong shape")
            difference = torch.abs(actual_cpu - expected_cpu)
            maximum_absolute = max(
                maximum_absolute,
                float(difference.max().item()) if difference.numel() else 0.0,
            )
            denominator = torch.clamp(torch.abs(expected_cpu), min=1e-12)
            relative = difference / denominator
            maximum_relative = max(
                maximum_relative,
                float(relative.max().item()) if relative.numel() else 0.0,
            )
            values_compared += expected_cpu.numel()
    passed = (
        maximum_absolute <= absolute_tolerance
        and maximum_relative <= relative_tolerance
    )
    return BackendPrecisionCertificate(
        ir.artifact_digest,
        plan.plan_digest,
        plan.backend,
        plan.dtype,
        len(requests),
        values_compared,
        maximum_absolute,
        maximum_relative,
        absolute_tolerance,
        relative_tolerance,
        passed,
    )


def execute_qualified(
    ir: CompiledResponseIR,
    plan: BackendPlan,
    certificate: BackendPrecisionCertificate,
    request: BackendRequest,
) -> BackendExecution:
    if not certificate.is_valid_for(ir, plan) or not certificate.passed:
        raise ValueError("backend plan lacks a valid passing precision certificate")
    return execute_backend_plan(plan, request)
