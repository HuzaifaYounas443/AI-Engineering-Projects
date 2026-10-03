"""PDF extractor."""
from pathlib import Path

import pdfplumber

from .base import BaseExtractor, ExtractedDocument


class PDFExtractor(BaseExtractor):
    """Handles .pdf files."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")

        content = "\n\n".join(pages)
        return ExtractedDocument(
            content=content,
            source_path=str(file_path),
            file_type=".pdf",
            metadata={"page_count": len(pages), "char_count": len(content)},
        )