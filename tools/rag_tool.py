class RAGTool:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def run(self, query):
        # Step 1: Retrieve
        docs = self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]

        # Step 2: Rerank
        state = {"query": query, "context": context}
        state = self.reranker(state)
        context = state["context"]

        # Step 3: Answer
        prompt = f"""
        Answer using the context. If not found, say I don't know.

        Context:
        {context}

        Question:
        {query}
        """

        return self.llm.invoke(prompt)
