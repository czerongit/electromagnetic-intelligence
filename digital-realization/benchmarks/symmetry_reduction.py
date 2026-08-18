from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource

from information_field.symmetry_field.reduction import compile_involutive_symmetry


def paired_source(pair_count: int):
    dimension = 2 * pair_count
    values = torch.repeat_interleave(
        torch.linspace(0.7, 1.7, pair_count, dtype=torch.float64), 2
    )
    coordinates = torch.arange(dimension, dtype=torch.int64)
    source = SparseRelationSource(
        dimension,
        dimension,
        coordinates,
        coordinates,
        values,
        torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    permutation = coordinates.clone()
    permutation[0::2] += 1
    permutation[1::2] -= 1
    symmetric = torch.zeros(dimension, dtype=torch.float64)
    antisymmetric = torch.zeros(dimension, dtype=torch.float64)
    symmetric[0::2] = symmetric[1::2] = 1.0
    antisymmetric[0::2] = 1.0
    antisymmetric[1::2] = -1.0
    port = torch.stack((symmetric, antisymmetric), dim=1)
    observation = torch.stack((symmetric, antisymmetric), dim=0)
    return source, port, observation, permutation


def _median_milliseconds(function, repetitions: int):
    samples = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return result, statistics.median(samples)


def benchmark_case(pair_count: int, repetitions: int) -> dict:
    source, port, observation, permutation = paired_source(pair_count)
    compile_matrix_free_relation_field(source, port, observation)
    compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    split, split_ms = _median_milliseconds(
        lambda: compile_involutive_symmetry(
            source, port, observation, permutation, permutation
        ),
        repetitions,
    )
    full, full_ms = _median_milliseconds(
        lambda: compile_matrix_free_relation_field(source, port, observation),
        repetitions,
    )
    incident = torch.tensor([0.31, -0.27], dtype=torch.float64)
    actual = split.respond_constant(incident, time=0.73, mass=1.4)
    expected = full.realization.respond_constant(incident, time=0.73, mass=1.4)
    return {
        "pair_count": pair_count,
        "ambient_dimension": source.quantity_dim,
        "source_nonzeros": source.nnz,
        "sector_count": split.accounting.sector_count,
        "sector_reachable_dimensions": split.accounting.sector_reachable_dimensions,
        "sector_minimal_dimensions": split.accounting.sector_minimal_dimensions,
        "split_factorized_actions": split.accounting.factorized_operator_applications,
        "unsplit_factorized_actions": full.accounting.factorized_operator_applications,
        "split_maximum_block_width": split.accounting.maximum_block_width,
        "unsplit_maximum_block_width": full.accounting.maximum_block_width,
        "symmetry_compile_ms": split_ms,
        "unsplit_compile_ms": full_ms,
        "compile_speedup": full_ms / split_ms,
        "response_absolute_error": float(torch.max(torch.abs(actual - expected)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="8,16,32,64")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        raise ValueError("repetitions must be positive")
    torch.set_num_threads(1)
    counts = tuple(int(value) for value in arguments.pairs.split(","))
    payload = {
        "contract": "exact involutive source-symmetry compilation",
        "timing": "median source-fixed compilation after one warmup",
        "results": [
            benchmark_case(count, arguments.repetitions) for count in counts
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
