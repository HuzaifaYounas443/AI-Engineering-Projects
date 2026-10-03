"""HTML extractor — pulls clean text from web pages."""
from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseExtractor, ExtractedDocument


class HTMLExtractor(BaseExtractor):
    """Handles .html / .htm files.

    Strips scripts, styles, nav, footer, and other non-content elements
    so we only embed meaningful page text.
    """

    # Tags whose content is never useful for retrieval
    DROP_TAGS = ["script", "style", "nav", "footer", "header",
                 "aside", "noscript", "iframe", "form"]

    def extract(self, file_path: Path) -> ExtractedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        # Capture page title before stripping anything
        title = soup.title.get_text(strip=True) if soup.title else ""

        # Remove non-content tags
        for tag in self.DROP_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Extract text with block-level newlines preserved
        text = soup.get_text(separator="\n", strip=True)

        # Collapse repeated blank lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        content = "\n".join(lines)

        return ExtractedDocument(
            content=content,
            source_path=str(file_path),
            file_type=file_path.suffix.lower(),
            metadata={
                "title": title,
                "char_count": len(content),
                "line_count": len(lines),
            },
        )