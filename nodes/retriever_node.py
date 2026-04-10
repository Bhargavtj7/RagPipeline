class RetrieverNode:
    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, state):
        query = state["query"]
        docs = self.retriever.invoke(query)

        state["context"] = [doc.page_content for doc in docs]
        return state
