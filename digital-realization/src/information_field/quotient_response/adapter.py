from __future__ import annotations

from dataclasses import dataclass

import torch

from .admission import QuotientCertificate, require_current_certificate
from .dynamics import CausalState, ExactModalResponse, compile_exact_modal, sparse_first_order_evolve
from .planner import CausalPlanDecision, choose_causal_plan
from .source import SparseIncidentBatch, SparseRelationSource, Tensor
from .static import CompiledStaticResponse, compile_static_response


@dataclass(frozen=True)
class CompiledCausalResponse:
    source: SparseRelationSource
    decision: CausalPlanDecision
    modal: ExactModalResponse | None

    def evolve_constant(
        self,
        relation_incident: Tensor,
        initial: CausalState,
        *,
        time: float,
        steps: int,
        mass: float = 1.0,
        calibration: float = 1.0,
    ) -> CausalState:
        if self.decision.plan == "exact-modal":
            if self.modal is None:
                raise RuntimeError("modal plan has no compiled factorization")
            return self.modal.evolve_constant(
                relation_incident,
                initial,
                time=time,
                mass=mass,
                calibration=calibration,
            )
        return sparse_first_order_evolve(
            self.source,
            relation_incident,
            initial,
            time=time,
            steps=steps,
            mass=mass,
            calibration=calibration,
        )


class QuotientResponseAdapter(torch.nn.Module):
    """Inference-only adapter for validated relation-coordinate responses."""

    def __init__(
        self,
        source: SparseRelationSource,
        observation: Tensor,
        admitted_features: Tensor,
    ) -> None:
        super().__init__()
        self.source = source
        self.observation = observation.to(source.device, source.dtype)
        self.static = compile_static_response(source, self.observation, admitted_features)

    def forward(self, incidents: SparseIncidentBatch) -> Tensor:
        if incidents.amplitudes.requires_grad:
            raise RuntimeError("the quotient-response realization is inference-only")
        return self.static.run(incidents)

    def verify_certificate(self, certificate: QuotientCertificate) -> None:
        require_current_certificate(self.source, certificate)

    def compile_causal(
        self,
        *,
        time_steps: int,
        expected_runs: int,
    ) -> CompiledCausalResponse:
        dense_rank = int(torch.linalg.matrix_rank(self.source.whitened_dense()).item())
        decision = choose_causal_plan(
            quantity_dim=self.source.quantity_dim,
            relation_dim=self.source.relation_dim,
            nonzeros=self.source.nnz,
            exact_rank=dense_rank,
            time_steps=time_steps,
            expected_runs=expected_runs,
        )
        modal = compile_exact_modal(self.source) if decision.plan == "exact-modal" else None
        return CompiledCausalResponse(self.source, decision, modal)
