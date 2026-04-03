"""Service for loading documents from CSV."""
import pandas as pd
from langchain_core.documents import Document
from typing import List


class DocumentLoader:
    """Responsible for loading and converting CSV data to LangChain documents."""

    def __init__(self, csv_path: str, review_col: str = "review", title_col: str = "title"):
        self.csv_path = csv_path
        self.review_col = review_col
        self.title_col = title_col

    def load(self) -> tuple[List[Document], List[str]]:
        """Load documents from CSV and return documents with their IDs."""
        df = pd.read_csv(self.csv_path)
        docs = []
        ids = []

        for index, row in df.iterrows():
            doc = Document(
                page_content=f"{row[self.review_col]} {row[self.title_col]}",
                metadata={
                    "rating": row.get("rating"),
                    "date": row.get("date")
                }
            )
            docs.append(doc)
            ids.append(str(index))

        return docs, ids
