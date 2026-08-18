"""Corpus-determined finite information fields."""

from .declarative import (
    DeclarativeRelationField,
    DeclarativeResponse,
    determine_declarative_field,
    evaluate_declarative_field,
)
from .wikipedia import WIKITEXT_FILES, load_wikitext_split

__all__ = [
    "DeclarativeRelationField",
    "DeclarativeResponse",
    "WIKITEXT_FILES",
    "determine_declarative_field",
    "evaluate_declarative_field",
    "load_wikitext_split",
]
