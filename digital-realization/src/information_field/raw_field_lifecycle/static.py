from __future__ import annotations

import hashlib

import torch

from information_field.quotient_response import CompiledStaticResponse, SparseRelationSource, StaticAccounting


def compile_identity_static_response(
    source: SparseRelationSource, admitted_features: torch.Tensor
) -> CompiledStaticResponse:
    """Compile observed D columns without materializing identity on H."""

    selected = torch.unique(
        admitted_features.to(device=source.device, dtype=torch.int64), sorted=True
    )
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= source.relation_dim
    ):
        raise ValueError("compiled feature is outside the relation carrier")
    lookup = torch.full(
        (source.relation_dim,), -1, dtype=torch.int64, device=source.device
    )
    if selected.numel():
        lookup[selected] = torch.arange(selected.numel(), device=source.device)
    columns = torch.zeros(
        (selected.numel(), source.quantity_dim),
        dtype=source.dtype,
        device=source.device,
    )
    selected_nonzeros = 0
    if selected.numel() and source.nnz:
        local = lookup[source.columns]
        admitted = local >= 0
        selected_nonzeros = int(admitted.sum().item())
        flat = local[admitted] * source.quantity_dim + source.rows[admitted]
        columns.view(-1).scatter_add_(0, flat, source.values[admitted])
    digest = hashlib.sha256(
        f"identity-observation:{source.quantity_dim}:{source.dtype}".encode()
    ).hexdigest()
    retained = sum(
        value.numel() * value.element_size()
        for value in (selected, lookup, columns)
    )
    return CompiledStaticResponse(
        source.digest,
        digest,
        source.relation_dim,
        selected,
        lookup,
        columns,
        StaticAccounting(
            int(selected.numel()),
            selected_nonzeros,
            retained,
            0,
            0,
            False,
        ),
    )
