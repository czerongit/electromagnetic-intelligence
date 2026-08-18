"""Minimal finite relation-coordinate response."""

import torch

from information_field import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)


source = SparseRelationSource.from_dense(
    torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
)
incident = SparseIncidentBatch(
    torch.tensor([[0], [1]], dtype=torch.int64),
    torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    torch.ones((2, 1), dtype=torch.bool),
)
response = compile_static_response(
    source,
    torch.eye(2, dtype=torch.float64),
    incident.admitted_features(),
)
print(response.run(incident))
