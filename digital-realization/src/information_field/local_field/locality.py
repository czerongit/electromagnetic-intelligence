from __future__ import annotations

from dataclasses import dataclass

import torch

from information_field.matrix_free_field import MatrixFreeCompilation, compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource


Tensor = torch.Tensor


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def quantity_components(source: SparseRelationSource) -> Tensor:
    union = _UnionFind(source.quantity_dim)
    rows = source.rows.detach().cpu().tolist()
    columns = source.columns.detach().cpu().tolist()
    by_relation: dict[int, list[int]] = {}
    for row, column in zip(rows, columns):
        by_relation.setdefault(column, []).append(row)
    for participants in by_relation.values():
        if not participants:
            continue
        anchor = participants[0]
        for participant in participants[1:]:
            union.union(anchor, participant)
    labels: dict[int, int] = {}
    result = []
    for node in range(source.quantity_dim):
        root = union.find(node)
        if root not in labels:
            labels[root] = len(labels)
        result.append(labels[root])
    return torch.tensor(result, dtype=torch.int64, device=source.device)


def quantity_adjacency(source: SparseRelationSource) -> tuple[tuple[int, ...], ...]:
    neighbors = [set((index,)) for index in range(source.quantity_dim)]
    rows = source.rows.detach().cpu().tolist()
    columns = source.columns.detach().cpu().tolist()
    by_relation: dict[int, list[int]] = {}
    for row, column in zip(rows, columns):
        by_relation.setdefault(column, []).append(row)
    for participants in by_relation.values():
        for row in participants:
            neighbors[row].update(participants)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _distances(
    adjacency: tuple[tuple[int, ...], ...],
    starts: tuple[int, ...],
    maximum: int,
) -> tuple[int, ...]:
    infinity = maximum + 1
    distance = [infinity] * len(adjacency)
    frontier = []
    for start in starts:
        if start < 0 or start >= len(adjacency):
            raise ValueError("support index is outside the quantity carrier")
        if distance[start] != 0:
            distance[start] = 0
            frontier.append(start)
    for depth in range(maximum):
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if distance[neighbor] > depth + 1:
                    distance[neighbor] = depth + 1
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(distance)


def jet_causal_nodes(
    source: SparseRelationSource,
    incident_support: tuple[int, ...],
    observation_support: tuple[int, ...],
    *,
    maximum_order: int,
) -> Tensor:
    if maximum_order < 0:
        raise ValueError("maximum jet order must be nonnegative")
    adjacency = quantity_adjacency(source)
    from_incident = _distances(adjacency, incident_support, maximum_order)
    from_observation = _distances(adjacency, observation_support, maximum_order)
    selected = [
        index
        for index, (left, right) in enumerate(zip(from_incident, from_observation))
        if left + right <= maximum_order
    ]
    return torch.tensor(selected, dtype=torch.int64, device=source.device)


@dataclass(frozen=True)
class ComponentAccounting:
    ambient_quantities: int
    ambient_relations: int
    component_count: int
    selected_components: int
    selected_quantities: int
    selected_relations: int
    selected_nonzeros: int
    zero_response: bool


@dataclass(frozen=True)
class ComponentRestrictedCompilation:
    compilation: MatrixFreeCompilation | None
    accounting: ComponentAccounting
    output_dimension: int
    source_digest: str

    def respond_constant(
        self,
        incident: Tensor,
        *,
        time: float,
        mass: float = 1.0,
    ) -> Tensor:
        if self.compilation is None:
            return torch.zeros(
                self.output_dimension,
                dtype=incident.dtype,
                device=incident.device,
            )
        return self.compilation.realization.respond_prepared_zero_past_constant(
            incident, time=time, mass=mass
        )


def compile_component_restricted(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    *,
    calibration: float = 1.0,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    tolerance: float = 1e-12,
) -> ComponentRestrictedCompilation:
    if relation_port.ndim != 2 or relation_port.shape[0] != source.relation_dim:
        raise ValueError("relation port has the wrong relation dimension")
    if observation.ndim != 2 or observation.shape[1] != source.quantity_dim:
        raise ValueError("observation has the wrong quantity dimension")
    n = source.quantity_dim
    position = (
        torch.empty((n, 0), dtype=source.dtype, device=source.device)
        if initial_position_port is None
        else initial_position_port.to(source.device, source.dtype)
    )
    velocity = (
        torch.empty((n, 0), dtype=source.dtype, device=source.device)
        if initial_velocity_port is None
        else initial_velocity_port.to(source.device, source.dtype)
    )
    labels = quantity_components(source)
    component_count = int(labels.max().item()) + 1 if labels.numel() else 0
    force = source.apply(relation_port.to(source.device, source.dtype).T).T
    seed_support = torch.any(torch.abs(force) > tolerance, dim=1)
    if position.shape[1]:
        seed_support |= torch.any(torch.abs(position) > tolerance, dim=1)
    if velocity.shape[1]:
        seed_support |= torch.any(torch.abs(velocity) > tolerance, dim=1)
    observation_support = torch.any(
        torch.abs(observation.to(source.device, source.dtype)) > tolerance,
        dim=0,
    )
    seed_components = set(labels[seed_support].detach().cpu().tolist())
    observed_components = set(labels[observation_support].detach().cpu().tolist())
    selected_component_values = sorted(seed_components & observed_components)
    if not selected_component_values:
        accounting = ComponentAccounting(
            n,
            source.relation_dim,
            component_count,
            0,
            0,
            0,
            0,
            True,
        )
        return ComponentRestrictedCompilation(
            None, accounting, observation.shape[0], source.digest
        )

    selected_components = torch.tensor(
        selected_component_values, dtype=torch.int64, device=source.device
    )
    quantity_mask = torch.isin(labels, selected_components)
    quantities = torch.nonzero(quantity_mask, as_tuple=False).flatten()
    entry_mask = quantity_mask[source.rows]
    relations = torch.unique(source.columns[entry_mask], sorted=True)
    row_lookup = torch.full((n,), -1, dtype=torch.int64, device=source.device)
    row_lookup[quantities] = torch.arange(quantities.numel(), device=source.device)
    column_lookup = torch.full(
        (source.relation_dim,), -1, dtype=torch.int64, device=source.device
    )
    column_lookup[relations] = torch.arange(relations.numel(), device=source.device)
    restricted = SparseRelationSource(
        int(quantities.numel()),
        int(relations.numel()),
        row_lookup[source.rows[entry_mask]],
        column_lookup[source.columns[entry_mask]],
        source.values[entry_mask],
        source.quantity_metric[quantities],
        source.relation_metric[relations],
    )
    compilation = compile_matrix_free_relation_field(
        restricted,
        relation_port.to(source.device, source.dtype)[relations],
        observation.to(source.device, source.dtype)[:, quantities],
        calibration=calibration,
        initial_position_port=(position[quantities] if position.shape[1] else None),
        initial_velocity_port=(velocity[quantities] if velocity.shape[1] else None),
        tolerance=tolerance,
    )
    accounting = ComponentAccounting(
        n,
        source.relation_dim,
        component_count,
        len(selected_component_values),
        int(quantities.numel()),
        int(relations.numel()),
        int(entry_mask.sum().item()),
        False,
    )
    return ComponentRestrictedCompilation(
        compilation, accounting, observation.shape[0], source.digest
    )
