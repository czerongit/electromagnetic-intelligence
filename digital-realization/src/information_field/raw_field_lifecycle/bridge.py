from __future__ import annotations

import math

import numpy as np
import torch

from information_field.quotient_response import (
    DeclarativeSourceBridge,
    SparseIncidentBatch,
    SparseRelationSource,
)


def lower_declarative_source(field) -> DeclarativeSourceBridge:
    """Lower corpus-determined D using one preallocated COO representation."""

    terms = tuple(field.terms)
    features = tuple(sorted(field.amplitudes_by_feature))
    feature_index = {feature: index for index, feature in enumerate(features)}
    nonzeros = field.nonzero_operator_entries
    rows = np.empty(nonzeros, dtype=np.int64)
    columns = np.empty(nonzeros, dtype=np.int64)
    values = np.empty(nonzeros, dtype=np.float64)
    cursor = 0
    for row, term in enumerate(terms):
        amplitudes = field.amplitudes_by_term[term]
        count = len(amplitudes)
        end = cursor + count
        rows[cursor:end] = row
        columns[cursor:end] = np.fromiter(
            (feature_index[feature] for feature in amplitudes),
            dtype=np.int64,
            count=count,
        )
        values[cursor:end] = np.fromiter(
            amplitudes.values(), dtype=np.float64, count=count
        )
        cursor = end
    if cursor != nonzeros:
        raise AssertionError("declarative COO count changed during lowering")
    source = SparseRelationSource(
        len(terms),
        len(features),
        torch.from_numpy(rows),
        torch.from_numpy(columns),
        torch.from_numpy(values),
        torch.ones(len(terms), dtype=torch.float64),
        torch.ones(len(features), dtype=torch.float64),
    )
    return DeclarativeSourceBridge(source, terms, features, feature_index)


def prepare_context_batch(
    bridge: DeclarativeSourceBridge,
    contexts: tuple[tuple[tuple[int, str], ...], ...],
) -> SparseIncidentBatch:
    """Restrict raw context incidents to the admitted relation carrier."""

    width = max((len(context) for context in contexts), default=0)
    indices = torch.zeros((len(contexts), width), dtype=torch.int64)
    amplitudes = torch.zeros((len(contexts), width), dtype=torch.float64)
    valid = torch.zeros((len(contexts), width), dtype=torch.bool)
    for row, context in enumerate(contexts):
        if not context:
            continue
        amplitude = 1.0 / math.sqrt(len(context))
        for column, feature in enumerate(context):
            local = bridge.feature_index.get(feature)
            if local is None:
                continue
            indices[row, column] = local
            amplitudes[row, column] = amplitude
            valid[row, column] = True
    return SparseIncidentBatch(indices, amplitudes, valid)
