from __future__ import annotations

import argparse
import json
import time

import torch

from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource

from information_field.incremental_field.update import try_reduced_update, update_or_recompile


def fixture(dimension: int):
    diagonal = torch.linspace(0.5, 2.0, dimension, dtype=torch.float64)
    old = SparseRelationSource.from_dense(torch.diag(diagonal))
    port = torch.eye(dimension, dtype=torch.float64)[:, :8]
    observation = torch.eye(dimension, dtype=torch.float64)[:4]
    compiled = compile_matrix_free_relation_field(old, port, observation)
    changed_diagonal = diagonal.clone()
    changed_diagonal[:8] *= torch.linspace(0.9, 1.1, 8, dtype=torch.float64)
    within = SparseRelationSource.from_dense(torch.diag(changed_diagonal))
    outside_diagonal = diagonal.clone()
    outside_diagonal[-1] *= 1.5
    hidden = SparseRelationSource.from_dense(torch.diag(outside_diagonal))
    expanded_dense = torch.diag(diagonal)
    expanded_dense[8, 0] = 0.25
    expanded = SparseRelationSource.from_dense(expanded_dense)
    return old, within, hidden, expanded, port, observation, compiled


def milliseconds(function):
    start = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - start) / 1_000_000.0


def benchmark_case(dimension: int) -> dict:
    old, within, hidden, expanded, port, observation, compiled = fixture(dimension)
    within_result, within_ms = milliseconds(
        lambda: try_reduced_update(compiled, old, within, port, observation)
    )
    full_result, full_ms = milliseconds(
        lambda: compile_matrix_free_relation_field(within, port, observation)
    )
    hidden_result, hidden_ms = milliseconds(
        lambda: try_reduced_update(compiled, old, hidden, port, observation)
    )
    fallback_result, fallback_ms = milliseconds(
        lambda: update_or_recompile(compiled, old, expanded, port, observation)
    )
    incident = torch.linspace(-0.5, 0.7, 8, dtype=torch.float64)
    updated = within_result.compilation.realization.respond_prepared_zero_past_constant(
        incident, time=0.6, mass=1.2
    )
    oracle = full_result.realization.respond_prepared_zero_past_constant(
        incident, time=0.6, mass=1.2
    )
    return {
        "ambient_dimension": dimension,
        "previous_reachable_dimension": compiled.accounting.reachable_dimension,
        "within_update_status": within_result.status,
        "within_update_ms": within_ms,
        "full_matrix_free_recompile_ms": full_ms,
        "within_update_speedup": full_ms / within_ms,
        "within_update_factor_applications": within_result.factorized_operator_applications,
        "full_recompile_factor_applications": full_result.accounting.factorized_operator_applications,
        "within_update_response_error": float(torch.max(torch.abs(updated - oracle)).item()),
        "hidden_edit_status": hidden_result.status,
        "hidden_edit_ms": hidden_ms,
        "hidden_edit_response_change": float(
            torch.max(
                torch.abs(
                    hidden_result.compilation.realization.respond_prepared_zero_past_constant(
                        incident, time=0.6, mass=1.2
                    )
                    - compiled.realization.respond_prepared_zero_past_constant(
                        incident, time=0.6, mass=1.2
                    )
                )
            ).item()
        ),
        "expanding_edit_status": fallback_result.status,
        "expanding_edit_total_ms": fallback_ms,
        "expanding_edit_outside_residual": fallback_result.invariance_outside_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="128,512,2048")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    payload = {
        "contract": "exact source-change update with mandatory global fallback",
        "results": [benchmark_case(dimension) for dimension in dimensions],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
