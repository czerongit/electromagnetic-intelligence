from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import torch

from information_field.causal_minimal import compile_minimal_realization
from information_field.observable_response import (
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from information_field.quotient_response import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)
from information_field.response_ir import (
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)

from .backend import (
    BackendKind,
    BackendRequest,
    backend_capability,
    choose_backend,
    lower_backend_plan,
    qualify_backend_plan,
)


def _temporal_ir():
    dtype = torch.float64
    operator = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=dtype))
    incident = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=dtype
    )
    observation = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=dtype
    )
    position = torch.tensor([[1.0], [0.0], [0.5]], dtype=dtype)
    velocity = torch.tensor([[0.0], [1.0], [-0.25]], dtype=dtype)
    realization = compile_minimal_realization(
        operator,
        incident,
        observation,
        initial_position_port=position,
        initial_velocity_port=velocity,
    )
    spectrum = compile_observable_spectrum(realization)
    return (
        lower_fixed_time(
            compile_fixed_time_green(realization, spectrum, time=0.73, mass=1.4)
        ),
        lower_sampled_times(
            compile_sampled_green(
                realization, spectrum, times=(0.2, 0.5, 1.1), mass=1.3
            )
        ),
        lower_grid_recurrence(
            compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
        ),
    )


def _static_ir_and_request():
    dtype = torch.float64
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=dtype)
    )
    observation = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=dtype)
    compiled = compile_static_response(
        source, observation, torch.tensor([0, 1], dtype=torch.int64)
    )
    incidents = SparseIncidentBatch(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.tensor([[0.3, -0.4], [0.7, 0.0]], dtype=dtype),
        torch.tensor([[True, True], [True, False]]),
    )
    prepared = compiled.prepare(incidents)
    request = BackendRequest(
        prepared.amplitudes,
        local_indices=prepared.local_indices,
        valid=prepared.valid,
    )
    return lower_static_response(compiled), request


def _record(certificate, plan) -> dict:
    payload = asdict(certificate)
    payload["backend"] = certificate.backend.value
    payload["dtype"] = str(certificate.dtype)
    payload["lowering"] = plan.lowering
    payload["fusion_claimed"] = plan.fusion_claimed
    payload["retained_bytes"] = plan.retained_bytes
    return payload


def run_qualification() -> dict:
    fixed, sampled, grid = _temporal_ir()
    static, static_request = _static_ir_and_request()
    fixed_request = BackendRequest(
        torch.tensor([0.2, -0.6], dtype=torch.float64),
        initial_position=torch.tensor([0.4], dtype=torch.float64),
        initial_velocity=torch.tensor([-0.3], dtype=torch.float64),
    )
    grid_request = BackendRequest(
        torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8], [0.1, -0.5]],
            dtype=torch.float64,
        )
    )
    contracts = (
        (static, static_request),
        (fixed, fixed_request),
        (sampled, fixed_request),
        (grid, grid_request),
    )
    results = []
    for backend, dtype, absolute, relative in (
        (BackendKind.CPU, torch.float64, 1e-12, 1e-12),
        (BackendKind.MPS, torch.float32, 2e-5, 2e-4),
    ):
        if not backend_capability(backend).supports(dtype):
            continue
        for ir, request in contracts:
            plan = lower_backend_plan(ir, backend, dtype)
            certificate = qualify_backend_plan(
                ir,
                plan,
                (request,),
                absolute_tolerance=absolute,
                relative_tolerance=relative,
            )
            result = _record(certificate, plan)
            result["contract"] = ir.contract.value
            results.append(result)
    capabilities = {}
    for backend in BackendKind:
        capability = backend_capability(backend)
        capabilities[backend.value] = {
            "available": capability.available,
            "reason": capability.reason,
            "supported_dtypes": [str(dtype) for dtype in capability.supported_dtypes],
        }
    cuda_decision = choose_backend(BackendKind.CUDA, torch.float32)
    return {
        "contract": "response backend precision qualification 90.1",
        "capabilities": capabilities,
        "cuda_request_decision": {
            "selected_backend": cuda_decision.selected_backend.value,
            "selected_dtype": str(cuda_decision.selected_dtype),
            "fallback_used": cuda_decision.fallback_used,
            "reason": cuda_decision.reason,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    payload = run_qualification()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
