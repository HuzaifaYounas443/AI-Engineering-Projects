"""Image OCR extractor."""
from pathlib import Path

import pytesseract
from PIL import Image

from .base import BaseExtractor, ExtractedDocument

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImageExtractor(BaseExtractor):
    """Handles .png / .jpg / .jpeg via Tesseract OCR."""

    def extract(self, file_path: Path) -> ExtractedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        image = Image.open(file_path).convert("L")
        text = pytesseract.image_to_string(image).strip()

        return ExtractedDocument(
            content=text,
            source_path=str(file_path),
            file_type=file_path.suffix.lower(),
            metadata={"char_count": len(text)},
        )