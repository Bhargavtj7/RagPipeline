from langsmith import traceable


class AgentNode:
    """Decision node that determines which tool to use."""

    def __init__(self, llm):
        self.llm = llm

    @traceable(name="agent_decision")
    def __call__(self, state):
        query = state["query"]

        prompt = (
            "You are a smart AI agent.\n"
            "Decide the best tool:\n"
            '- "general" \u2192 greetings/simple chat\n'
            '- "rag" \u2192 for internal knowledge (PDF, CSV, HTML)\n'
            '- "web" \u2192 for latest/current info\n'
            '- "rewrite" \u2192 if query is unclear\n'
            "Think step-by-step, then output ONLY one word:\n"
            "general / rag / web / rewrite\n"
            f"Query: {query}"
        )

        decision = self.llm.invoke(prompt).strip().lower()

        # Safety guard (VERY important in real systems)
        valid_tools = {"general", "rag", "web", "rewrite"}
        if decision not in valid_tools:
            decision = "general"  # fallback

        state["tool"] = decision
        return state
