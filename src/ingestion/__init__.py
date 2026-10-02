"""Ingestion package — public API."""
from .chunker import chunk_document, chunk_text
from .embedder import Embedder
from .extractors.base import ExtractedDocument
from .indexer import QdrantIndexer
from .processor import IngestionProcessor, UnsupportedFileTypeError

__all__ = [
    "IngestionProcessor",
    "UnsupportedFileTypeError",
    "ExtractedDocument",
    "chunk_document",
    "chunk_text",
    "Embedder",
    "QdrantIndexer",
]