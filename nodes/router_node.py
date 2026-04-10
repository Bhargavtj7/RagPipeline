class RouterNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        query = state["query"]

        prompt = f"""
        Classify the query:
        If greeting or casual → general
        Else → rag

        Query: {query}
        Answer: only 'general' or 'rag'
        """

        decision = self.llm.invoke(prompt).strip().lower()

        state["is_general"] = decision == "general"
        return state
