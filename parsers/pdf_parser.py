from langchain_core.documents import Document
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from .base_parser import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF files using Marker library."""

    def __init__(self):
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def parse(self, file_path):
        """Parse PDF file and extract text."""
        try:
            rendered = self.converter(file_path)
            text, _, _ = text_from_rendered(rendered)

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                    },
                )
            ]

        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(f"Error parsing PDF: {e}") from e
