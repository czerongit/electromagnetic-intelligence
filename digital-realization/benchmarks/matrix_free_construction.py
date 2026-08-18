from __future__ import annotations

import argparse
import json
import time

import torch

from information_field.causal_minimal import compile_relation_field
from information_field.quotient_response import SparseRelationSource

from information_field.matrix_free_field.compiler import compile_matrix_free_relation_field


def make_diagonal_source(dimension: int):
    rows = torch.arange(dimension, dtype=torch.int64)
    columns = rows.clone()
    values = torch.linspace(0.5, 2.0, dimension, dtype=torch.float64)
    metric = torch.ones(dimension, dtype=torch.float64)
    source = SparseRelationSource(
        dimension,
        dimension,
        rows,
        columns,
        values,
        metric,
        metric.clone(),
    )
    relation_port = torch.eye(dimension, dtype=torch.float64)[:, :8]
    observation = torch.eye(dimension, dtype=torch.float64)[:4]
    return source, relation_port, observation


def benchmark_case(dimension: int, *, run_dense: bool) -> dict:
    source, relation_port, observation = make_diagonal_source(dimension)
    started = time.perf_counter_ns()
    sparse = compile_matrix_free_relation_field(
        source, relation_port, observation, calibration=0.8
    )
    matrix_free_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    dense_ms = None
    response_error = None
    if run_dense:
        started = time.perf_counter_ns()
        dense = compile_relation_field(
            source, relation_port, observation, calibration=0.8
        )
        dense_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        incident = torch.linspace(-0.5, 0.7, 8, dtype=torch.float64)
        sparse_response = sparse.realization.respond_prepared_zero_past_constant(
            incident, time=0.6, mass=1.2
        )
        dense_response = dense.respond_prepared_zero_past_constant(
            incident, time=0.6, mass=1.2
        )
        response_error = float(
            torch.max(torch.abs(sparse_response - dense_response)).item()
        )

    scalar_bytes = torch.tensor([], dtype=torch.float64).element_size()
    dense_ambient_bytes = 2 * dimension * dimension * scalar_bytes
    workspace_bytes = (
        dimension
        * (
            sparse.accounting.reachable_dimension
            + sparse.accounting.maximum_block_width
            + relation_port.shape[1]
        )
        * scalar_bytes
    )
    return {
        "ambient_dimension": dimension,
        "source_nonzeros": source.nnz,
        "reachable_dimension": sparse.accounting.reachable_dimension,
        "minimal_dimension": sparse.accounting.minimal_dimension,
        "matrix_free_compile_ms": matrix_free_ms,
        "dense_compile_ms": dense_ms,
        "compile_speedup": (
            None if dense_ms is None else dense_ms / matrix_free_ms
        ),
        "response_absolute_error": response_error,
        "factorized_operator_applications": sparse.accounting.factorized_operator_applications,
        "d_applications": sparse.accounting.d_applications,
        "adjoint_applications": sparse.accounting.adjoint_applications,
        "source_bytes": sparse.accounting.source_bytes,
        "matrix_free_workspace_estimate_bytes": workspace_bytes,
        "dense_ambient_operator_estimate_bytes": dense_ambient_bytes,
        "ambient_byte_avoidance": dense_ambient_bytes
        / max(1, sparse.accounting.source_bytes + workspace_bytes),
        "dense_relation_operator_materialized": sparse.accounting.dense_relation_operator_materialized,
        "dense_intrinsic_operator_materialized": sparse.accounting.dense_intrinsic_operator_materialized,
        "maximum_markov_residual": sparse.realization.certificate.maximum_markov_residual,
        "reachable_invariance_residual": sparse.realization.certificate.reachable_invariance_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="128,512,2048")
    parser.add_argument("--dense-limit", type=int, default=2048)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    dimensions = tuple(int(value) for value in arguments.dimensions.split(","))
    results = [
        benchmark_case(dimension, run_dense=dimension <= arguments.dense_limit)
        for dimension in dimensions
    ]
    payload = {
        "contract": "source-fixed exact causal-carrier compilation",
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
