from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownParser:
    """Parser for Markdown files."""

    def parse(self, file_path):
        """Parse Markdown file and split by headers."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

        docs = splitter.split_text(text)
        return docs
