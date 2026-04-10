class AnswerNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        query = state["query"]
        context = state["context"]

        prompt = f"""
        Answer the question using ONLY the context.

        Context:
        {context}

        Question:
        {query}
        """

        state["answer"] = self.llm.invoke(prompt)
        return state
