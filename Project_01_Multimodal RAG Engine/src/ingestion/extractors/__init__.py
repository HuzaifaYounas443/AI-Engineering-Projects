"""Extractor registry — maps file-type keys to instances."""
from .text_extractor import TextExtractor
from .pdf_extractor import PDFExtractor
from .image_extractor import ImageExtractor
from .docx_extractor import DocxExtractor
from .html_extractor import HTMLExtractor

EXTRACTOR_REGISTRY = {
    "text": TextExtractor(),
    "pdf": PDFExtractor(),
    "image": ImageExtractor(),
    "docx": DocxExtractor(),
    "html": HTMLExtractor(),
}