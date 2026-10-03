"""Text / Markdown extractor."""
from pathlib import Path

from .base import BaseExtractor, ExtractedDocument


class TextExtractor(BaseExtractor):
    """Handles .txt / .md files."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return ExtractedDocument(
            content=content,
            source_path=str(file_path),
            file_type=file_path.suffix.lower(),
            metadata={"char_count": len(content)},
        )