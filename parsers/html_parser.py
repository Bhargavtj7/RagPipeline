from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .base_parser import BaseParser


class HTMLParser(BaseParser):
    """Parser for HTML files."""

    def parse(self, file_path):
        """Parse HTML file and extract text."""
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        text = soup.get_text()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "html",
                },
            )
        ]
