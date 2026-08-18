from __future__ import annotations

import argparse
import json
import time

import torch

from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource

from information_field.local_field.locality import compile_component_restricted


def disconnected_chains(component_count: int, component_size: int = 16):
    quantity_dim = component_count * component_size
    relation_dim = component_count * (component_size - 1)
    rows = []
    columns = []
    values = []
    relation = 0
    for component in range(component_count):
        start = component * component_size
        for offset in range(component_size - 1):
            rows.extend((start + offset, start + offset + 1))
            columns.extend((relation, relation))
            values.extend((1.0, -1.0))
            relation += 1
    source = SparseRelationSource(
        quantity_dim,
        relation_dim,
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int64),
        torch.tensor(values, dtype=torch.float64),
        torch.ones(quantity_dim, dtype=torch.float64),
        torch.ones(relation_dim, dtype=torch.float64),
    )
    port = torch.zeros((relation_dim, 4), dtype=torch.float64)
    port[:4] = torch.eye(4, dtype=torch.float64)
    observation = torch.zeros((4, quantity_dim), dtype=torch.float64)
    observation[:, :4] = torch.eye(4, dtype=torch.float64)
    return source, port, observation


def timed(function):
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def benchmark_case(component_count: int) -> dict:
    source, port, observation = disconnected_chains(component_count)
    local, local_ms = timed(
        lambda: compile_component_restricted(source, port, observation)
    )
    global_compiled, global_ms = timed(
        lambda: compile_matrix_free_relation_field(source, port, observation)
    )
    incident = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
    actual = local.respond_constant(incident, time=0.6, mass=1.2)
    expected = global_compiled.realization.respond_prepared_zero_past_constant(
        incident, time=0.6, mass=1.2
    )
    return {
        "component_count": component_count,
        "ambient_quantities": source.quantity_dim,
        "ambient_relations": source.relation_dim,
        "ambient_nonzeros": source.nnz,
        "selected_quantities": local.accounting.selected_quantities,
        "selected_relations": local.accounting.selected_relations,
        "selected_nonzeros": local.accounting.selected_nonzeros,
        "component_compile_ms": local_ms,
        "global_matrix_free_compile_ms": global_ms,
        "compile_speedup": global_ms / local_ms,
        "response_absolute_error": float(torch.max(torch.abs(actual - expected)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", default="8,32,128,512")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    counts = tuple(int(value) for value in arguments.components.split(","))
    payload = {
        "contract": "exact disconnected-component causal compilation",
        "component_size": 16,
        "results": [benchmark_case(count) for count in counts],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
