"""Matched retrieval inputs and relation-coordinate response."""

from dataclasses import replace

import torch

from .contracts import AttentionInput
from .problems import BindingProblem, make_binding_problem, score_binding_result
from .relation import (
    RelationIncidents,
    RelationNativeSource,
    canonical_relation_incidents,
    dense_quotient_response,
    determine_canonical_fixture,
    determine_relation_source,
)


def move_problem(
    problem: BindingProblem,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BindingProblem:
    """Move one retrieval problem without changing its associations."""

    target = torch.device(device)

    def move(value: torch.Tensor) -> torch.Tensor:
        if value.is_floating_point():
            return value.to(device=target, dtype=dtype)
        return value.to(device=target)

    current = problem.attention_input
    return replace(
        problem,
        attention_input=AttentionInput(
            move(current.prefix_values),
            move(current.incident_values),
            move(current.prefix_positions),
            move(current.incident_positions),
            move(current.prefix_valid),
            move(current.incident_valid),
        ),
        key_bases=move(problem.key_bases),
        payload_bases=move(problem.payload_bases),
        target_payload_indices=move(problem.target_payload_indices),
        binding_permutations=move(problem.binding_permutations),
        basis_fingerprints=move(problem.basis_fingerprints),
    )


__all__ = [
    "AttentionInput",
    "BindingProblem",
    "RelationIncidents",
    "RelationNativeSource",
    "canonical_relation_incidents",
    "dense_quotient_response",
    "determine_canonical_fixture",
    "determine_relation_source",
    "make_binding_problem",
    "move_problem",
    "score_binding_result",
]
