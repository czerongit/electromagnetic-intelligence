from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from information_field.native_response_kernels import (
    MPSResponseKernels,
    NativeKernelExecution,
    NativeKernelRecord,
)
from information_field.profiled_response import PreparedBackendRequest
from information_field.response_backends import BackendKind, BackendPlan
from information_field.response_ir import IRExecution, ResponseContract


class WideMPSResponseKernels:
    def __init__(self) -> None:
        path = Path(__file__).parents[1] / "native" / "wide_response.metal"
        source = path.read_text(encoding="utf-8")
        self.source_digest = hashlib.sha256(source.encode()).hexdigest()
        self.library = torch.mps.compile_shader(source)

    def execute(
        self,
        plan: BackendPlan,
        prepared: PreparedBackendRequest,
    ) -> NativeKernelExecution:
        if plan.backend is not BackendKind.MPS or plan.dtype is not torch.float32:
            raise ValueError("wide Metal kernels require an MPS FP32 plan")
        if prepared.plan_digest != plan.plan_digest or prepared.contract is not plan.contract:
            raise ValueError("prepared request belongs to another backend plan")
        if plan.contract is not ResponseContract.REGULAR_GRID:
            raise ValueError("wide Metal execution applies only to regular-grid response")
        modes = plan.tensor("cosine").numel()
        if max(modes, plan.output_dimension) <= MPSResponseKernels.MAX_THREADGROUP_WIDTH:
            raise ValueError("use the one-threadgroup executor for this width")
        incidents = prepared.incident
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
        for step in range(steps):
            self.library.modal_step_parallel(
                position,
                velocity,
                incidents[step],
                plan.tensor("cosine"),
                plan.tensor("sine_over_omega"),
                plan.tensor("negative_omega_sine"),
                plan.tensor("force_position"),
                plan.tensor("force_velocity"),
                plan.tensor("modal_incident"),
                modes,
                plan.incident_width,
                threads=[modes],
            )
            self.library.modal_readout_parallel(
                output[step],
                position,
                plan.tensor("modal_observation"),
                modes,
                plan.output_dimension,
                threads=[plan.output_dimension],
            )
        dispatches = 2 * steps
        return NativeKernelExecution(
            IRExecution(output, position, velocity),
            NativeKernelRecord(
                plan.plan_digest,
                BackendKind.MPS,
                plan.contract,
                "modal_history_multidispatch",
                self.source_digest,
                dispatches,
                False,
                False,
            ),
        )
