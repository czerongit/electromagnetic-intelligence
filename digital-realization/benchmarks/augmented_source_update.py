from __future__ import annotations

import argparse
import json
import time

import torch

from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource

from information_field.augmented_update.update import try_augmented_update


def sparse_diagonal(dimension: int, *, add_coupling: bool):
    indices = torch.arange(dimension, dtype=torch.int64)
    rows = indices
    columns = indices.clone()
    values = torch.linspace(0.5, 2.0, dimension, dtype=torch.float64)
    if add_coupling:
        rows = torch.cat((rows, torch.tensor([8], dtype=torch.int64)))
        columns = torch.cat((columns, torch.tensor([0], dtype=torch.int64)))
        values = torch.cat((values, torch.tensor([0.25], dtype=torch.float64)))
    metric = torch.ones(dimension, dtype=torch.float64)
    return SparseRelationSource(
        dimension,
        dimension,
        rows,
        columns,
        values,
        metric,
        metric.clone(),
    )


def timed(function):
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def benchmark_case(dimension: int) -> dict:
    if dimension < 16:
        raise ValueError("dimension must be at least 16")
    old = sparse_diagonal(dimension, add_coupling=False)
    new = sparse_diagonal(dimension, add_coupling=True)
    port = torch.eye(dimension, dtype=torch.float64)[:, :8]
    observation = torch.eye(dimension, dtype=torch.float64)[:9]
    previous = compile_matrix_free_relation_field(old, port, observation)
    augmented, augmented_ms = timed(
        lambda: try_augmented_update(previous, old, new, port, observation)
    )
    full, full_ms = timed(
        lambda: compile_matrix_free_relation_field(new, port, observation)
    )
    incident = torch.linspace(-0.5, 0.7, 8, dtype=torch.float64)
    actual = augmented.compilation.realization.respond_prepared_zero_past_constant(
        incident, time=0.6, mass=1.2
    )
    expected = full.realization.respond_prepared_zero_past_constant(
        incident, time=0.6, mass=1.2
    )
    return {
        "ambient_dimension": dimension,
        "source_nonzeros": new.nnz,
        "old_reachable_dimension": augmented.old_reachable_dimension,
        "augmented_dimension": augmented.augmented_dimension,
        "final_reachable_dimension": augmented.final_reachable_dimension,
        "added_dimensions": augmented.added_dimensions,
        "augmented_status": augmented.status,
        "augmented_update_ms": augmented_ms,
        "full_matrix_free_recompile_ms": full_ms,
        "update_speedup": full_ms / augmented_ms,
        "augmented_operator_applications": augmented.total_operator_applications,
        "full_operator_applications": full.accounting.factorized_operator_applications,
        "response_absolute_error": float(torch.max(torch.abs(actual - expected)).item()),
        "maximum_markov_residual": augmented.compilation.realization.certificate.maximum_markov_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="128,512,2048,8192")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    payload = {
        "contract": "exact one-direction reachable-carrier augmentation",
        "results": [benchmark_case(dimension) for dimension in dimensions],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
