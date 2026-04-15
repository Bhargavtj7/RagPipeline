from langsmith import traceable


class RAGTool:
    """Tool for RAG-based retrieval and answering."""

    def __init__(self, retriever, reranker, llm, web_tool=None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.web_tool = web_tool  # optional

    @traceable(name="rag_tool")
    def run(self, query):
        # Step 1: Retrieve documents
        docs = self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]

        # Step 2: Rerank
        state = {"query": query, "context": context}
        state = self.reranker(state)
        context = state["context"]

        # Step 3: Fallback to web if context is weak
        if len(context) < 2 and self.web_tool:
            web_result = self.web_tool.run(query)
            context.append(web_result)

        # Step 4: LLM call
        prompt = (
            "Answer using the context.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question:\n{query}"
        )

        return self.llm.invoke(prompt)
