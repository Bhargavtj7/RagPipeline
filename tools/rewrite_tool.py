class RewriteTool:
    """Tool for rewriting unclear queries."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, query):
        """Rewrite query to improve retrieval."""
        prompt = "Rewrite this query to improve retrieval:\n" "\n" f"{query}"
        return self.llm.invoke(prompt)
