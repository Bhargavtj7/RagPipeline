import pandas as pd
from langchain_core.documents import Document

from .base_parser import BaseParser


class CSVParser(BaseParser):
    def parse(self, file_path):
        df = pd.read_csv(file_path)

        docs = []
        for i, row in df.iterrows():
            docs.append(
                Document(
                    page_content=str(row.to_dict()),
                    metadata={"source": file_path, "row": i, "type": "csv"},
                )
            )

        return docs
