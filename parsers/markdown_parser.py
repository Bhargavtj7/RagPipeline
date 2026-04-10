from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownParser:
    def parse(self, file_path):
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
