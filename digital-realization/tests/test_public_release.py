from pathlib import Path

import pytest
import torch

import information_field
from information_field.geometric_observation.wikipedia import load_wikitext_split
from information_field.reduction import select_static_executor
from information_field.retrieval import (
    dense_quotient_response,
    determine_canonical_fixture,
    make_binding_problem,
)


def test_curated_package_api_executes_static_response():
    source = information_field.SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    )
    incident = information_field.SparseIncidentBatch(
        torch.tensor([[0], [1]], dtype=torch.int64),
        torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        torch.ones((2, 1), dtype=torch.bool),
    )
    response = information_field.compile_static_response(
        source,
        torch.eye(2, dtype=torch.float64),
        incident.admitted_features(),
    )
    assert torch.equal(
        response.run(incident),
        torch.tensor([[3.0, 0.0], [0.0, 8.0]], dtype=torch.float64),
    )


def test_retrieval_source_is_target_independent_and_selects_columns():
    problem = make_binding_problem(
        batch_size=2,
        binding_count=4,
        prefix_length=64,
        seed=7601,
        split="test",
    )
    source = determine_canonical_fixture(problem.attention_input, 4)
    response = dense_quotient_response(source)
    scores = torch.abs(
        torch.einsum("bqd,bkd->bqk", response, problem.payload_bases)
    )
    assert torch.equal(torch.argmax(scores, dim=-1), problem.target_payload_indices)
    selections = select_static_executor(source.observed_columns, support=1)
    assert all(selection.executor == "columns" for selection in selections)


def test_corpus_loader_never_downloads_implicitly(tmp_path: Path):
    with pytest.raises(ValueError, match="path is required"):
        load_wikitext_split("train")
    missing = tmp_path / "train.parquet"
    with pytest.raises(FileNotFoundError):
        load_wikitext_split("train", missing)
