from __future__ import annotations

from dataclasses import dataclass

import torch

from .source import SparseIncident, SparseRelationSource


@dataclass(frozen=True)
class DeclarativeSourceBridge:
    source: SparseRelationSource
    terms: tuple[str, ...]
    features: tuple[tuple[int, str], ...]
    feature_index: dict[tuple[int, str], int]

    def incident(self, context: tuple[tuple[int, str], ...]) -> SparseIncident:
        if not context:
            return SparseIncident(
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=self.source.dtype),
            )
        missing = tuple(feature for feature in context if feature not in self.feature_index)
        if missing:
            raise ValueError(f"context contains unadmitted relation features: {missing}")
        amplitude = len(context) ** -0.5
        return SparseIncident(
            torch.tensor(
                [self.feature_index[feature] for feature in context],
                dtype=torch.int64,
                device=self.source.device,
            ),
            torch.full(
                (len(context),),
                amplitude,
                dtype=self.source.dtype,
                device=self.source.device,
            ),
        )


def from_declarative_field(field) -> DeclarativeSourceBridge:
    """Lower a finite declarative relation operator without materializing it densely."""

    terms = tuple(field.terms)
    features = tuple(sorted(field.amplitudes_by_feature))
    feature_index = {feature: index for index, feature in enumerate(features)}
    rows = []
    columns = []
    values = []
    for row, term in enumerate(terms):
        for feature, amplitude in field.amplitudes_by_term[term].items():
            rows.append(row)
            columns.append(feature_index[feature])
            values.append(amplitude)
    dtype = torch.float64
    source = SparseRelationSource(
        len(terms),
        len(features),
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int64),
        torch.tensor(values, dtype=dtype),
        torch.ones(len(terms), dtype=dtype),
        torch.ones(len(features), dtype=dtype),
    )
    return DeclarativeSourceBridge(source, terms, features, feature_index)
