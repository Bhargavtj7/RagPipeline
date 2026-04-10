class CombinedRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        all_docs = []

        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            all_docs.extend(docs)

        return all_docs
