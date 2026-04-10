class GeneralNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        query = state["query"]
        state["answer"] = self.llm.invoke(query)
        return state
