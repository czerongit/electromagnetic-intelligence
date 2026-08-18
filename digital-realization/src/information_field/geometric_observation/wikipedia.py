from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd
from pathlib import Path

import pyarrow.parquet as parquet


WIKITEXT_ROOT = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-v1"
)
WIKITEXT_FILES = {
    "train": (
        "train-00000-of-00001.parquet",
        "dfc27e4360c639dc1fba1e403bfffd53af4a5c75d5363b5724d49bf12d07cce6",
    ),
    "validation": (
        "validation-00000-of-00001.parquet",
        "717de9a0c1c0b0b1dfdd8f1e6ad8a30ece618bbde81f5da8207277547d324215",
    ),
    "test": (
        "test-00000-of-00001.parquet",
        "e6b3913da714b63a60a571698b20ff15441fb015783ea1b5285f707d4f2f00a9",
    ),
}
TOKEN = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class WikipediaGeometryAnalysis:
    source_sha256: str | None
    paragraphs: int
    token_occurrences: int
    vocabulary: int
    exact_sectors: int
    nonsingleton_exact_sectors: int
    terms_in_nonsingleton_sectors: int
    largest_exact_sector: int
    minimum_neighbor_occurrences: int
    nearest_neighbors: dict[str, tuple[tuple[str, float], ...]]
    exact_sector_examples: tuple[tuple[str, ...], ...]


def load_wikitext_split(
    split: str, path: Path | None = None
) -> tuple[tuple[str, ...], str]:
    """Load a caller-supplied WikiText-2 split and verify its bytes.

    Files are never downloaded implicitly.  ``path`` must identify the
    corresponding parquet file named in :data:`WIKITEXT_FILES`.
    """

    if split not in WIKITEXT_FILES:
        raise ValueError(f"unsupported WikiText-2 split: {split}")
    filename, expected_digest = WIKITEXT_FILES[split]
    if path is None:
        raise ValueError(
            f"path is required for WikiText-2 {split}; expected {filename}"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"unexpected WikiText-2 {split} SHA-256: {digest}")
    texts = tuple(
        value
        for value in parquet.read_table(path, columns=["text"])["text"].to_pylist()
        if value and value.strip()
    )
    return texts, digest


def load_wikitext_train(path: Path | None = None) -> tuple[tuple[str, ...], str]:
    return load_wikitext_split("train", path)


def _context_counts(
    paragraphs: tuple[str, ...], radius: int
) -> tuple[dict[str, Counter[tuple[int, str]]], Counter[str], int]:
    contexts: dict[str, Counter[tuple[int, str]]] = defaultdict(Counter)
    occurrences: Counter[str] = Counter()
    total = 0
    for paragraph in paragraphs:
        tokens = [match.group(0).casefold() for match in TOKEN.finditer(paragraph)]
        if not tokens:
            continue
        padded = ["<bos>"] * radius + tokens + ["<eos>"] * radius
        for index, term in enumerate(tokens, start=radius):
            occurrences[term] += 1
            total += 1
            for offset in range(-radius, radius + 1):
                if offset:
                    contexts[term][(offset, padded[index + offset])] += 1
    return dict(contexts), occurrences, total


def _exact_signature(
    counts: Counter[tuple[int, str]],
) -> tuple[tuple[int, str, int], ...]:
    divisor = reduce(gcd, counts.values())
    return tuple(
        sorted(
            (offset, neighbor, count // divisor)
            for (offset, neighbor), count in counts.items()
        )
    )


def _hellinger_coordinates(
    counts: Counter[tuple[int, str]], total: int
) -> dict[tuple[int, str], float]:
    return {feature: math.sqrt(count / total) for feature, count in counts.items()}


def _affinity(
    left: dict[tuple[int, str], float], right: dict[tuple[int, str], float]
) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


def analyze_wikipedia_geometry(
    paragraphs: tuple[str, ...],
    *,
    source_sha256: str | None = None,
    radius: int = 3,
    minimum_neighbor_occurrences: int = 50,
    probes: tuple[str, ...] = (
        "album",
        "city",
        "film",
        "king",
        "river",
        "species",
        "war",
        "woman",
    ),
) -> WikipediaGeometryAnalysis:
    """Analyze exact sectors and the unthresholded context-response geometry.

    Exact sectors require identical normalized contextual measures. Continuous
    neighbors use Hellinger affinity between those measures. Neither path fits
    weights, requests a sector count, or assigns a semantic label.
    """

    if radius < 1:
        raise ValueError("radius must be positive")
    contexts, occurrences, token_occurrences = _context_counts(paragraphs, radius)
    exact_groups: dict[tuple[tuple[int, str, int], ...], list[str]] = defaultdict(list)
    for term, counts in contexts.items():
        exact_groups[_exact_signature(counts)].append(term)
    groups = tuple(tuple(sorted(group)) for group in exact_groups.values())
    nonsingletons = tuple(
        sorted(
            (group for group in groups if len(group) > 1),
            key=lambda group: (-len(group), group),
        )
    )

    eligible = tuple(
        term for term, count in occurrences.items() if count >= minimum_neighbor_occurrences
    )
    coordinates = {
        term: _hellinger_coordinates(contexts[term], occurrences[term] * 2 * radius)
        for term in eligible
    }
    nearest: dict[str, tuple[tuple[str, float], ...]] = {}
    for probe in probes:
        term = probe.casefold()
        if term not in coordinates:
            continue
        ranked = sorted(
            (
                (candidate, _affinity(coordinates[term], vector))
                for candidate, vector in coordinates.items()
                if candidate != term
            ),
            key=lambda item: (-item[1], item[0]),
        )
        nearest[term] = tuple(
            (candidate, round(score, 6)) for candidate, score in ranked[:8]
        )

    return WikipediaGeometryAnalysis(
        source_sha256=source_sha256,
        paragraphs=len(paragraphs),
        token_occurrences=token_occurrences,
        vocabulary=len(contexts),
        exact_sectors=len(groups),
        nonsingleton_exact_sectors=len(nonsingletons),
        terms_in_nonsingleton_sectors=sum(map(len, nonsingletons)),
        largest_exact_sector=max(map(len, groups), default=0),
        minimum_neighbor_occurrences=minimum_neighbor_occurrences,
        nearest_neighbors=nearest,
        exact_sector_examples=nonsingletons[:20],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--minimum-occurrences", type=int, default=50)
    arguments = parser.parse_args()
    paragraphs, digest = load_wikitext_train(arguments.parquet)
    result = analyze_wikipedia_geometry(
        paragraphs,
        source_sha256=digest,
        radius=arguments.radius,
        minimum_neighbor_occurrences=arguments.minimum_occurrences,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
