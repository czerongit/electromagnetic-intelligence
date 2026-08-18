from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch

from .wikipedia import TOKEN, _context_counts, load_wikitext_split


Feature = tuple[int, str]


@dataclass(frozen=True)
class DeclarativeResponse:
    context: tuple[Feature, ...]
    ranked_terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DeclarativeRelationField:
    """Finite corpus-determined relation operator D: G -> H."""

    radius: int
    normalization: str
    terms: tuple[str, ...]
    occurrences: dict[str, int]
    amplitudes_by_term: dict[str, dict[Feature, float]]
    amplitudes_by_feature: dict[Feature, tuple[tuple[str, float], ...]]

    @property
    def relation_feature_count(self) -> int:
        return len(self.amplitudes_by_feature)

    @property
    def nonzero_operator_entries(self) -> int:
        return sum(len(values) for values in self.amplitudes_by_term.values())

    def incident(self, unit: tuple[str, ...], position: int) -> tuple[Feature, ...]:
        if not 0 <= position < len(unit):
            raise IndexError(position)
        padded = ("<bos>",) * self.radius + unit + ("<eos>",) * self.radius
        center = position + self.radius
        return tuple(
            (offset, padded[center + offset])
            for offset in range(-self.radius, self.radius + 1)
            if offset
        )

    @staticmethod
    def relation_incident(
        context: tuple[Feature, ...],
    ) -> dict[Feature, float]:
        """Construct the unit relation-carrier incident determined by context."""

        if not context:
            return {}
        amplitude = 1.0 / math.sqrt(len(context))
        return {feature: amplitude for feature in context}

    def source_covector(
        self, context: tuple[Feature, ...]
    ) -> dict[str, float]:
        """Return coefficients of the induced configuration-source covector.

        In the finite Euclidean carriers these coefficients are the Riesz
        representative D q_C. They are automatically in range(D).
        """

        return self.induce_source_covector(self.relation_incident(context))

    def induce_source_covector(
        self, incident: dict[Feature, float]
    ) -> dict[str, float]:
        """Apply D to an already determined relation-carrier incident."""

        scores: dict[str, float] = defaultdict(float)
        for feature, incident_amplitude in incident.items():
            for term, amplitude in self.amplitudes_by_feature.get(feature, ()):
                scores[term] += amplitude * incident_amplitude
        return dict(scores)

    def respond(
        self, context: tuple[Feature, ...], *, top_k: int = 5
    ) -> DeclarativeResponse:
        """Observe the sign-adjusted response induced by the raw-text covector."""

        if not context:
            return DeclarativeResponse(context, ())
        scores = self.source_covector(context)
        ranked = tuple(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        )
        return DeclarativeResponse(context, ranked)

    def dense_operator(self) -> tuple[torch.Tensor, tuple[Feature, ...]]:
        """Materialize D for finite identity tests, not full-corpus execution."""

        features = tuple(sorted(self.amplitudes_by_feature))
        feature_index = {feature: index for index, feature in enumerate(features)}
        operator = torch.zeros(
            (len(self.terms), len(features)), dtype=torch.float64
        )
        for term_index, term in enumerate(self.terms):
            for feature, amplitude in self.amplitudes_by_term[term].items():
                operator[term_index, feature_index[feature]] = amplitude
        return operator, features


@dataclass(frozen=True)
class DeclarativeWikipediaEvaluation:
    train_sha256: str
    evaluation_split: str
    evaluation_sha256: str
    train_records: int
    evaluation_records: int
    operator_normalization: str
    candidate_terms: int
    relation_features: int
    nonzero_operator_entries: int
    held_out_queries: int
    target_response_coverage: float
    top_1_accuracy: float
    top_5_accuracy: float
    mean_reciprocal_rank: float
    frequency_top_1_accuracy: float
    frequency_top_5_accuracy: float
    examples: tuple[dict[str, object], ...]


def determine_declarative_field(
    paragraphs: tuple[str, ...],
    *,
    radius: int = 3,
    minimum_occurrences: int = 50,
    normalization: str = "joint",
) -> DeclarativeRelationField:
    """Determine D from declarations only; no query or target is accepted."""

    if normalization not in {"conditional", "joint"}:
        raise ValueError("normalization must be conditional or joint")
    contexts, occurrence_counts, _ = _context_counts(paragraphs, radius)
    terms = tuple(
        sorted(
            term
            for term, count in occurrence_counts.items()
            if count >= minimum_occurrences
        )
    )
    amplitudes_by_term: dict[str, dict[Feature, float]] = {}
    inverse: dict[Feature, list[tuple[str, float]]] = defaultdict(list)
    joint_denominator = sum(occurrence_counts[term] for term in terms) * 2 * radius
    for term in terms:
        denominator = (
            occurrence_counts[term] * 2 * radius
            if normalization == "conditional"
            else joint_denominator
        )
        amplitudes = {
            feature: math.sqrt(count / denominator)
            for feature, count in contexts[term].items()
        }
        amplitudes_by_term[term] = amplitudes
        for feature, amplitude in amplitudes.items():
            inverse[feature].append((term, amplitude))
    return DeclarativeRelationField(
        radius=radius,
        normalization=normalization,
        terms=terms,
        occurrences={term: occurrence_counts[term] for term in terms},
        amplitudes_by_term=amplitudes_by_term,
        amplitudes_by_feature={
            feature: tuple(values) for feature, values in inverse.items()
        },
    )


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN.finditer(text))


def _held_out_occurrences(
    paragraphs: tuple[str, ...],
    admitted_terms: frozenset[str],
    maximum: int,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    selected: list[tuple[int, tuple[str, ...], int]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        unit = _tokenize(paragraph)
        for position, term in enumerate(unit):
            if term not in admitted_terms:
                continue
            key = int.from_bytes(
                hashlib.blake2b(
                    f"{paragraph_index}:{position}".encode(), digest_size=8
                ).digest(),
                "big",
            )
            selected.append((key, unit, position))
    selected.sort(key=lambda item: item[0])
    return tuple((unit, position) for _, unit, position in selected[:maximum])


def evaluate_declarative_field(
    field: DeclarativeRelationField,
    validation: tuple[str, ...],
    *,
    maximum_queries: int = 2_000,
) -> dict[str, object]:
    """Score frozen D; concealed targets never participate in determination."""

    occurrences = _held_out_occurrences(
        validation, frozenset(field.terms), maximum_queries
    )
    frequency_order = tuple(
        sorted(field.terms, key=lambda term: (-field.occurrences[term], term))
    )
    frequency_top_1 = frequency_order[:1]
    frequency_top_5 = frozenset(frequency_order[:5])
    top_1_correct = 0
    top_5_correct = 0
    frequency_1_correct = 0
    frequency_5_correct = 0
    supported = 0
    reciprocal_rank = 0.0
    examples: list[dict[str, object]] = []

    for unit, position in occurrences:
        target = unit[position]
        context = field.incident(unit, position)
        scores = field.source_covector(context)
        target_score = scores.get(target, 0.0)
        supported += int(target_score > 0.0)
        rank = 1 + sum(
            score > target_score or (score == target_score and term < target)
            for term in field.terms
            if (score := scores.get(term, 0.0)) >= target_score
        )
        reciprocal_rank += 1.0 / rank
        ranked = tuple(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5]
        )
        predicted = ranked[0][0] if ranked else frequency_top_1[0]
        predicted_five = frozenset(term for term, _ in ranked)
        top_1_correct += int(predicted == target)
        top_5_correct += int(target in predicted_five)
        frequency_1_correct += int(target in frequency_top_1)
        frequency_5_correct += int(target in frequency_top_5)
        if len(examples) < 12:
            window_start = max(0, position - field.radius)
            window_end = min(len(unit), position + field.radius + 1)
            window = list(unit[window_start:window_end])
            window[position - window_start] = "<mask>"
            examples.append(
                {
                    "context": " ".join(window),
                    "target": target,
                    "rank": rank,
                    "response": tuple(
                        (term, round(score, 6)) for term, score in ranked
                    ),
                }
            )

    count = len(occurrences)
    if not count:
        raise ValueError("validation contains no admitted held-out occurrences")
    return {
        "held_out_queries": count,
        "target_response_coverage": supported / count,
        "top_1_accuracy": top_1_correct / count,
        "top_5_accuracy": top_5_correct / count,
        "mean_reciprocal_rank": reciprocal_rank / count,
        "frequency_top_1_accuracy": frequency_1_correct / count,
        "frequency_top_5_accuracy": frequency_5_correct / count,
        "examples": tuple(examples),
    }


def evaluate_declarative_wikipedia(
    train: tuple[str, ...],
    evaluation_corpus: tuple[str, ...],
    *,
    train_sha256: str,
    evaluation_split: str,
    evaluation_sha256: str,
    radius: int = 3,
    minimum_occurrences: int = 50,
    maximum_queries: int = 2_000,
    normalization: str = "joint",
) -> DeclarativeWikipediaEvaluation:
    field = determine_declarative_field(
        train,
        radius=radius,
        minimum_occurrences=minimum_occurrences,
        normalization=normalization,
    )
    evaluation = evaluate_declarative_field(
        field, evaluation_corpus, maximum_queries=maximum_queries
    )
    return DeclarativeWikipediaEvaluation(
        train_sha256=train_sha256,
        evaluation_split=evaluation_split,
        evaluation_sha256=evaluation_sha256,
        train_records=len(train),
        evaluation_records=len(evaluation_corpus),
        operator_normalization=field.normalization,
        candidate_terms=len(field.terms),
        relation_features=field.relation_feature_count,
        nonzero_operator_entries=field.nonzero_operator_entries,
        **evaluation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--minimum-occurrences", type=int, default=50)
    parser.add_argument("--maximum-queries", type=int, default=2_000)
    parser.add_argument(
        "--normalization", choices=("joint", "conditional"), default="joint"
    )
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"), default="validation"
    )
    arguments = parser.parse_args()
    train, train_digest = load_wikitext_split("train")
    evaluation_corpus, evaluation_digest = load_wikitext_split(
        arguments.evaluation_split
    )
    result = evaluate_declarative_wikipedia(
        train,
        evaluation_corpus,
        train_sha256=train_digest,
        evaluation_split=arguments.evaluation_split,
        evaluation_sha256=evaluation_digest,
        radius=arguments.radius,
        minimum_occurrences=arguments.minimum_occurrences,
        maximum_queries=arguments.maximum_queries,
        normalization=arguments.normalization,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
