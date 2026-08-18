from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from information_field.native_response_kernels import NativeKernelExecution, NativeKernelRecord
from information_field.profiled_response import PreparedBackendRequest
from information_field.response_backends import BackendKind, BackendPlan
from information_field.response_ir import IRExecution, ResponseContract


class WideCUDAResponseKernels:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        path = Path(__file__).parents[1] / "native" / "wide_response_cuda.cu"
        source = path.read_bytes()
        self.source_digest = hashlib.sha256(source).hexdigest()
        self.module = load(
            name="information_field_wide_response_cuda_93",
            sources=[str(path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )

    def execute(
        self,
        plan: BackendPlan,
        prepared: PreparedBackendRequest,
    ) -> NativeKernelExecution:
        if plan.backend is not BackendKind.CUDA or plan.dtype is not torch.float32:
            raise ValueError("wide CUDA kernels require a CUDA FP32 plan")
        if prepared.plan_digest != plan.plan_digest or prepared.contract is not plan.contract:
            raise ValueError("prepared request belongs to another backend plan")
        if plan.contract is not ResponseContract.REGULAR_GRID:
            raise ValueError("wide CUDA execution applies only to regular-grid response")
        modes = plan.tensor("cosine").numel()
        if max(modes, plan.output_dimension) <= 1024:
            raise ValueError("use the one-block executor for this width")
        incidents = prepared.incident
        position = (
            torch.zeros(modes, device="cuda", dtype=torch.float32)
            if prepared.state_position is None
            else prepared.state_position
        )
        velocity = (
            torch.zeros(modes, device="cuda", dtype=torch.float32)
            if prepared.state_velocity is None
            else prepared.state_velocity
        )
        output, final_position, final_velocity = self.module.modal_history(
            incidents,
            plan.tensor("cosine"),
            plan.tensor("sine_over_omega"),
            plan.tensor("negative_omega_sine"),
            plan.tensor("force_position"),
            plan.tensor("force_velocity"),
            plan.tensor("modal_incident"),
            plan.tensor("modal_observation"),
            position,
            velocity,
        )
        dispatches = 2 * incidents.shape[0]
        return NativeKernelExecution(
            IRExecution(output, final_position, final_velocity),
            NativeKernelRecord(
                plan.plan_digest,
                BackendKind.CUDA,
                plan.contract,
                "modal_history_multidispatch_cuda",
                self.source_digest,
                dispatches,
                False,
                False,
            ),
        )
