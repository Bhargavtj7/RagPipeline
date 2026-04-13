class ToolNode:
    def __init__(self, tools):
        self.tools = tools

    def __call__(self, state):
        tool_name = state["tool"]
        query = state["query"]

        tool = self.tools[tool_name]
        result = tool.run(query)

        # If rewrite → update query
        if tool_name == "rewrite":
            state["query"] = result
        else:
            state["answer"] = result

        return state
