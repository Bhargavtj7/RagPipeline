from langsmith import traceable


class ToolNode:
    """Node that executes the selected tool."""

    def __init__(self, tools):
        self.tools = tools

    @traceable(name="tool_execution")
    def __call__(self, state):
        tool_name = state["tool"]
        query = state["query"]

        # Safety check (prevents crashes)
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        result = tool.run(query)

        # Handle rewrite vs other tools
        if tool_name == "rewrite":
            state["query"] = result
        else:
            state["answer"] = result

        return state
