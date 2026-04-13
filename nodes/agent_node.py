class AgentNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        query = state["query"]

        prompt = f"""
        You are an AI agent with 3 tools:

        1. general → for greetings/simple chat
        2. rag → for knowledge-based questions
        3. rewrite → if query is unclear

        Decide the best tool.

        Query: {query}

        Answer ONLY:
        general / rag / rewrite
        """

        decision = self.llm.invoke(prompt).strip().lower()
        state["tool"] = decision
        return state
