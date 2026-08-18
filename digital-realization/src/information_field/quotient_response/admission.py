from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .source import SparseIncident, SparseRelationSource, Tensor


@dataclass(frozen=True)
class QuotientCertificate:
    status: Literal["determined", "ambiguous", "unsupported"]
    source_digest: str
    compatible_count: int
    representative: SparseIncident | None
    induced_source: Tensor | None
    maximum_image_difference: float

    @property
    def accepted(self) -> bool:
        return self.status == "determined"


def certify_compatible_incidents(
    source: SparseRelationSource,
    incidents: tuple[SparseIncident, ...],
    *,
    tolerance: float = 1e-10,
) -> QuotientCertificate:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not incidents:
        return QuotientCertificate(
            "unsupported", source.digest, 0, None, None, 0.0
        )

    dense = []
    try:
        for incident in incidents:
            dense.append(incident.to(source.device, source.dtype).dense(source.relation_dim))
    except ValueError:
        return QuotientCertificate(
            "unsupported", source.digest, len(incidents), None, None, 0.0
        )
    images = tuple(source.apply(value) for value in dense)
    reference = images[0]
    maximum = max(
        (float(torch.max(torch.abs(value - reference)).item()) for value in images[1:]),
        default=0.0,
    )
    if maximum > tolerance:
        return QuotientCertificate(
            "ambiguous", source.digest, len(incidents), None, None, maximum
        )
    return QuotientCertificate(
        "determined",
        source.digest,
        len(incidents),
        incidents[0],
        reference,
        maximum,
    )


def require_current_certificate(
    source: SparseRelationSource, certificate: QuotientCertificate
) -> None:
    if not certificate.accepted:
        raise ValueError(f"question covector is {certificate.status}")
    if certificate.source_digest != source.digest:
        raise ValueError("question certificate was determined for another source")
