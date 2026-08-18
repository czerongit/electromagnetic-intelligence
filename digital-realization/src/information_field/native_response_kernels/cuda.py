from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from information_field.profiled_response import PreparedBackendRequest
from information_field.response_backends import BackendKind, BackendPlan
from information_field.response_ir import IRExecution, ResponseContract

from .metal import NativeKernelExecution, NativeKernelRecord


class CUDAResponseKernels:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        path = Path(__file__).parents[1] / "native" / "response_cuda.cu"
        source = path.read_bytes()
        self.source_digest = hashlib.sha256(source).hexdigest()
        self.module = load(
            name="information_field_response_cuda_92",
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
            raise ValueError("native CUDA kernels require a CUDA FP32 plan")
        if prepared.plan_digest != plan.plan_digest or prepared.contract is not plan.contract:
            raise ValueError("prepared request belongs to another backend plan")
        with torch.inference_mode():
            if plan.contract is ResponseContract.STATIC_COLUMNS:
                output = self.module.weighted_columns(
                    plan.tensor("observed_columns"),
                    prepared.local_indices,
                    prepared.local_weights,
                    prepared.offsets,
                )
                return self._result(plan, "weighted_columns_cuda", IRExecution(output))
            if plan.contract in {
                ResponseContract.FIXED_TIME,
                ResponseContract.SAMPLED_TIMES,
            }:
                output = self.module.packed_mv(
                    plan.tensor("packed_map"), prepared.packed_coordinates
                )
                if plan.contract is ResponseContract.SAMPLED_TIMES:
                    output = output.reshape(-1, plan.output_dimension)
                return self._result(plan, "packed_mv_cuda", IRExecution(output))
            modes = plan.tensor("cosine").numel()
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
                prepared.incident,
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
            return self._result(
                plan,
                "modal_history_cuda",
                IRExecution(output, final_position, final_velocity),
                dispatches=1 if prepared.incident.shape[0] else 0,
                serial=bool(
                    max(modes, plan.output_dimension) > 1024
                    and prepared.incident.shape[0]
                ),
            )

    def _result(
        self,
        plan: BackendPlan,
        kernel: str,
        result: IRExecution,
        *,
        dispatches: int = 1,
        serial: bool = False,
    ) -> NativeKernelExecution:
        return NativeKernelExecution(
            result,
            NativeKernelRecord(
                plan.plan_digest,
                BackendKind.CUDA,
                plan.contract,
                kernel,
                self.source_digest,
                dispatches,
                serial,
                dispatches == 1,
            ),
        )
