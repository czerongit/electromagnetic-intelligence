from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CausalPlanDecision:
    plan: Literal["sparse-first-order", "exact-modal"]
    sparse_work: int
    modal_work: int
    sparse_retained_scalars: int
    modal_retained_scalars: int
    reason: str


def choose_causal_plan(
    *,
    quantity_dim: int,
    relation_dim: int,
    nonzeros: int,
    exact_rank: int,
    time_steps: int,
    expected_runs: int,
) -> CausalPlanDecision:
    if min(quantity_dim, relation_dim, time_steps, expected_runs) < 1:
        raise ValueError("plan dimensions, steps, and runs must be positive")
    if not 0 <= exact_rank <= min(quantity_dim, relation_dim):
        raise ValueError("exact rank is invalid")
    sparse_work = expected_runs * time_steps * 2 * nonzeros
    modal_work = exact_rank * (quantity_dim + relation_dim) + expected_runs * time_steps * exact_rank
    sparse_retained = 3 * nonzeros + quantity_dim + relation_dim
    modal_retained = exact_rank * (quantity_dim + relation_dim + 1)
    modal_wins = (
        exact_rank > 0
        and modal_work < sparse_work
        and modal_retained < sparse_retained
    )
    return CausalPlanDecision(
        "exact-modal" if modal_wins else "sparse-first-order",
        sparse_work,
        modal_work,
        sparse_retained,
        modal_retained,
        "exact modal work and storage are both lower"
        if modal_wins
        else "sparse factorized execution wins ties or one resource comparison",
    )
