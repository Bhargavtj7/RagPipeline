class AnswerNode:
    """Node that generates answers using LLM and context."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        query = state["query"]
        context = state["context"]

        prompt = (
            "Answer the question using ONLY the context.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question:\n{query}"
        )

        state["answer"] = self.llm.invoke(prompt)
        return state
