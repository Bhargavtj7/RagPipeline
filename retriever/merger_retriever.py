class CombinedRetriever:
    """Retriever that combines results from multiple retrievers."""

    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        """Invoke all retrievers and combine results."""
        all_docs = []

        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            all_docs.extend(docs)

        return all_docs
