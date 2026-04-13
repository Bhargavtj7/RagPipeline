class RewriteTool:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query):
        prompt = f"""
        Rewrite this query to improve retrieval:

        {query}
        """
        return self.llm.invoke(prompt)
