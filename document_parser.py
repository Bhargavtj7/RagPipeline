import re
from pathlib import Path

from parsers.csv_parser import CSVParser
from parsers.html_parser import HTMLParser
from parsers.markdown_parser import MarkdownParser
from parsers.pdf_parser import PDFParser
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    def __init__(self):
        self.parsers = {
            "pdf": PDFParser(),
            "csv": CSVParser(),
            "html": HTMLParser(),
            "md": MarkdownParser(),
        }
        # Regex for supported files
        self.pattern = re.compile(
            r".*\.(pdf|csv|html|md)$",
            re.IGNORECASE,
        )

    def parse_directory(self, directory_path):
        all_docs = []

        for file in Path(directory_path).rglob("*"):
            if file.is_file() and self.pattern.match(str(file)):
                ext = file.suffix.lower().replace(".", "")

                try:
                    logger.info("Processing file: %s", file)

                    parser = self.parsers.get(ext)

                    if parser:
                        parsed_docs = parser.parse(str(file))
                        all_docs.extend(parsed_docs)

                        parsed_count = len(parsed_docs)
                        logger.info(
                            "Parsed %d docs from %s",
                            parsed_count,
                            file,
                        )

                except Exception as e:
                    logger.error("Error processing %s: %s", file, e)

        logger.info("Total parsed docs: %d", len(all_docs))
        return all_docs
