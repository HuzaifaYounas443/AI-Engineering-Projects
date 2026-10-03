"""Base extractor interface — every extractor returns an ExtractedDocument."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedDocument:
    """Uniform document representation."""
    content: str
    source_path: str
    file_type: str
    metadata: dict = field(default_factory=dict)


class BaseExtractor(ABC):
    """All extractors implement extract()."""

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        ...