from __future__ import annotations

from dataclasses import dataclass

import torch

from .source import SparseRelationSource, Tensor
from .static import CompiledStaticResponse
from .static import StaticAccounting, observation_digest


@dataclass(frozen=True)
class InvalidationDecision:
    changed_features: Tensor
    invalidated_compiled_features: Tensor
    static_compilation_valid: bool
    modal_compilation_valid: bool
    reason: str


def source_change_invalidation(
    old: SparseRelationSource,
    new: SparseRelationSource,
    compiled: CompiledStaticResponse,
    *,
    tolerance: float = 0.0,
) -> InvalidationDecision:
    changed = old.changed_features(new, tolerance=tolerance)
    compiled_cpu = compiled.selected_features.detach().cpu()
    affected = compiled_cpu[torch.isin(compiled_cpu, changed.cpu())]
    metric_changed = not (
        old.quantity_metric.shape == new.quantity_metric.shape
        and old.relation_metric.shape == new.relation_metric.shape
        and torch.equal(old.quantity_metric.cpu(), new.quantity_metric.cpu())
        and torch.equal(old.relation_metric.cpu(), new.relation_metric.cpu())
    )
    rank_changed = False
    if (old.quantity_dim, old.relation_dim) == (new.quantity_dim, new.relation_dim):
        rank_changed = int(torch.linalg.matrix_rank(old.dense_operator()).item()) != int(
            torch.linalg.matrix_rank(new.dense_operator()).item()
        )
    else:
        rank_changed = True
    static_valid = not metric_changed and affected.numel() == 0
    modal_valid = changed.numel() == 0 and not metric_changed and not rank_changed
    reasons = []
    if affected.numel():
        reasons.append("compiled sparse columns changed")
    if metric_changed:
        reasons.append("carrier metric changed")
    if rank_changed:
        reasons.append("source rank changed")
    if changed.numel() and not reasons:
        reasons.append("uncompiled source columns changed")
    return InvalidationDecision(
        changed,
        affected,
        static_valid,
        modal_valid,
        "; ".join(reasons) if reasons else "query-only or identical source",
    )


def update_static_compilation(
    compiled: CompiledStaticResponse,
    old: SparseRelationSource,
    new: SparseRelationSource,
    observation: Tensor,
    *,
    tolerance: float = 0.0,
) -> CompiledStaticResponse:
    if old.relation_dim != new.relation_dim or old.quantity_dim != new.quantity_dim:
        raise ValueError("carrier change requires full static recompilation")
    if not torch.equal(old.quantity_metric.cpu(), new.quantity_metric.cpu()) or not torch.equal(
        old.relation_metric.cpu(), new.relation_metric.cpu()
    ):
        raise ValueError("metric change requires full static recompilation")
    observation = observation.to(new.device, new.dtype)
    if observation_digest(observation) != compiled.observation_digest:
        raise ValueError("observation change requires full static recompilation")

    selected = compiled.selected_features.to(new.device)
    all_changed = old.changed_features(new, tolerance=tolerance).to(new.device)
    changed = selected[torch.isin(selected, all_changed)]
    updated_columns = compiled.observed_columns.to(new.device, new.dtype).clone()
    if changed.numel():
        selected_lookup = torch.full(
            (new.relation_dim,), -1, dtype=torch.int64, device=new.device
        )
        selected_lookup[selected] = torch.arange(selected.numel(), device=new.device)
        local = selected_lookup[changed]
        updated_columns[local] = new.observed_columns(observation, changed)
    selected_nnz = int(torch.isin(new.columns, selected).sum().item()) if selected.numel() else 0
    retained = sum(
        value.numel() * value.element_size()
        for value in (selected, compiled.feature_lookup, updated_columns)
    )
    return CompiledStaticResponse(
        new.digest,
        compiled.observation_digest,
        new.relation_dim,
        selected,
        compiled.feature_lookup.to(new.device),
        updated_columns,
        StaticAccounting(
            int(selected.numel()),
            selected_nnz,
            retained,
            0,
            0,
            False,
        ),
    )
