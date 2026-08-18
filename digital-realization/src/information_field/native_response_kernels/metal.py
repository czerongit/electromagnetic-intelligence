from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from information_field.profiled_response import PreparedBackendRequest
from information_field.response_backends import BackendKind, BackendPlan
from information_field.response_ir import IRExecution, ResponseContract


Tensor = torch.Tensor


@dataclass(frozen=True)
class NativeKernelRecord:
    plan_digest: str
    backend: BackendKind
    contract: ResponseContract
    kernel_name: str
    source_digest: str
    dispatches: int
    serial_device_execution: bool
    fusion_certified: bool


@dataclass(frozen=True)
class NativeKernelExecution:
    result: IRExecution
    record: NativeKernelRecord


class MPSResponseKernels:
    MAX_THREADGROUP_WIDTH = 1024

    def __init__(self, *, recurrence_mode: str = "auto") -> None:
        if recurrence_mode not in {"auto", "serial"}:
            raise ValueError("recurrence mode must be auto or serial")
        path = Path(__file__).parents[1] / "native" / "response.metal"
        source = path.read_text(encoding="utf-8")
        self.source_digest = hashlib.sha256(source.encode()).hexdigest()
        self.library = torch.mps.compile_shader(source)
        self.recurrence_mode = recurrence_mode

    def execute(
        self,
        plan: BackendPlan,
        prepared: PreparedBackendRequest,
    ) -> NativeKernelExecution:
        if plan.backend is not BackendKind.MPS or plan.dtype is not torch.float32:
            raise ValueError("native Metal kernels require an MPS FP32 plan")
        if prepared.plan_digest != plan.plan_digest or prepared.contract is not plan.contract:
            raise ValueError("prepared request belongs to another backend plan")
        if plan.contract is ResponseContract.STATIC_COLUMNS:
            batch = prepared.offsets.numel() - 1
            output = torch.empty(
                (batch, plan.output_dimension), device="mps", dtype=torch.float32
            )
            self.library.weighted_columns(
                output,
                plan.tensor("observed_columns"),
                prepared.local_indices,
                prepared.local_weights,
                prepared.offsets,
                plan.output_dimension,
            )
            return self._result(plan, "weighted_columns", IRExecution(output), False)
        if plan.contract in {ResponseContract.FIXED_TIME, ResponseContract.SAMPLED_TIMES}:
            rows = plan.tensor("packed_map").shape[0]
            output = torch.empty(rows, device="mps", dtype=torch.float32)
            self.library.packed_mv(
                output,
                plan.tensor("packed_map"),
                prepared.packed_coordinates,
                plan.tensor("packed_map").shape[1],
            )
            if plan.contract is ResponseContract.SAMPLED_TIMES:
                output = output.reshape(-1, plan.output_dimension)
            return self._result(plan, "packed_mv", IRExecution(output), False)
        incidents = prepared.incident
        modes = plan.tensor("cosine").numel()
        steps = incidents.shape[0]
        position = (
            torch.zeros(modes, device="mps", dtype=torch.float32)
            if prepared.state_position is None
            else prepared.state_position.clone()
        )
        velocity = (
            torch.zeros(modes, device="mps", dtype=torch.float32)
            if prepared.state_velocity is None
            else prepared.state_velocity.clone()
        )
        output = torch.empty(
            (steps, plan.output_dimension), device="mps", dtype=torch.float32
        )
        thread_count = max(modes, plan.output_dimension)
        use_threadgroup = (
            self.recurrence_mode == "auto"
            and thread_count <= self.MAX_THREADGROUP_WIDTH
        )
        if steps and use_threadgroup:
            self.library.modal_history_threadgroup(
                output,
                position,
                velocity,
                incidents,
                plan.tensor("cosine"),
                plan.tensor("sine_over_omega"),
                plan.tensor("negative_omega_sine"),
                plan.tensor("force_position"),
                plan.tensor("force_velocity"),
                plan.tensor("modal_incident"),
                plan.tensor("modal_observation"),
                modes,
                plan.incident_width,
                plan.output_dimension,
                steps,
                threads=[thread_count],
                group_size=[thread_count],
            )
        elif steps:
            self.library.modal_history_serial(
                output,
                position,
                velocity,
                incidents,
                plan.tensor("cosine"),
                plan.tensor("sine_over_omega"),
                plan.tensor("negative_omega_sine"),
                plan.tensor("force_position"),
                plan.tensor("force_velocity"),
                plan.tensor("modal_incident"),
                plan.tensor("modal_observation"),
                modes,
                plan.incident_width,
                plan.output_dimension,
                steps,
            )
        return self._result(
            plan,
            "modal_history_threadgroup" if use_threadgroup else "modal_history_serial",
            IRExecution(output, position, velocity),
            not use_threadgroup,
            dispatches=1 if steps else 0,
        )

    def _result(
        self,
        plan: BackendPlan,
        kernel: str,
        result: IRExecution,
        serial: bool,
        *,
        dispatches: int = 1,
    ) -> NativeKernelExecution:
        return NativeKernelExecution(
            result,
            NativeKernelRecord(
                plan.plan_digest,
                BackendKind.MPS,
                plan.contract,
                kernel,
                self.source_digest,
                dispatches,
                serial,
                dispatches == 1,
            ),
        )
