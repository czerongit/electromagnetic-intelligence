from __future__ import annotations

import json
from pathlib import Path

import torch

from information_field.geometric_observation import determine_declarative_field
from information_field.quotient_response import compile_static_response, from_declarative_field
from information_field.raw_field_lifecycle import (
    compile_identity_static_response,
    lower_declarative_source,
    prepare_context_batch,
)


def declarations():
    return (
        "Red fox crosses the hill.",
        "Blue wolf crosses the hill.",
        "Red fox enters the cave.",
        "Blue wolf enters the cave.",
    )


def test_preallocated_declarative_lowering_preserves_sparse_source():
    field = determine_declarative_field(
        declarations(), radius=2, minimum_occurrences=1, normalization="joint"
    )
    expected = from_declarative_field(field)
    actual = lower_declarative_source(field)
    assert actual.terms == expected.terms
    assert actual.features == expected.features
    assert torch.equal(actual.source.rows, expected.source.rows)
    assert torch.equal(actual.source.columns, expected.source.columns)
    assert torch.equal(actual.source.values, expected.source.values)
    assert actual.source.digest == expected.source.digest


def test_identity_compiler_matches_general_observation_without_dense_requirement():
    field = determine_declarative_field(
        declarations(), radius=2, minimum_occurrences=1, normalization="joint"
    )
    bridge = lower_declarative_source(field)
    selected = torch.tensor([0, 2, 3], dtype=torch.int64)
    actual = compile_identity_static_response(bridge.source, selected)
    expected = compile_static_response(
        bridge.source,
        torch.eye(bridge.source.quantity_dim, dtype=torch.float64),
        selected,
    )
    assert torch.equal(actual.selected_features, expected.selected_features)
    assert torch.equal(actual.feature_lookup, expected.feature_lookup)
    assert torch.equal(actual.observed_columns, expected.observed_columns)
    assert not actual.accounting.dense_observation_operator_materialized


def test_context_restriction_preserves_source_covector():
    field = determine_declarative_field(
        declarations(), radius=2, minimum_occurrences=1, normalization="joint"
    )
    bridge = lower_declarative_source(field)
    context = ((-2, "red"), (-1, "fox"), (1, "the"), (2, "unseen"))
    batch = prepare_context_batch(bridge, (context,))
    induced = bridge.source.apply(batch.dense(bridge.source.relation_dim)[0])
    expected_sparse = field.source_covector(context)
    expected = torch.tensor(
        [expected_sparse.get(term, 0.0) for term in bridge.terms],
        dtype=torch.float64,
    )
    assert torch.allclose(induced, expected)
    assert int(batch.valid.sum()) == 3


def test_recorded_wikitext_lifecycle_covers_complete_boundaries():
    payload = json.loads(
        (
            Path(__file__).parents[1]
            / "results"
            / "published"
            / "wikitext-lifecycle.json"
        ).read_text()
    )
    assert payload["queries"] == 512
    assert payload["operator_nonzeros"] == 3_296_117
    assert payload["phase_seconds"]["determine_source"] > 0
    assert payload["phase_seconds"]["prepare_compiled_incidents"] > 0
    assert payload["source_replacement"]["source_digest_changed"]
    assert payload["source_replacement"]["artifact_digest_changed"]
    assert payload["source_replacement"]["direct_oracle_seconds"] > 0
    assert payload["quality"]["top_1_accuracy"] > 0.10
    assert payload["quality"]["top_5_accuracy"] > 0.28
    assert {record["backend"] for record in payload["backends"]} == {"cpu", "mps"}
    assert all(record["maximum_absolute_error"] < 1e-7 for record in payload["backends"])
    assert payload["boundaries"]["generative_language_model"] == "not established"
