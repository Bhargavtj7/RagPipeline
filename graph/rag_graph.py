from langgraph.graph import END, StateGraph

from graph.state import GraphState
from nodes.agent_node import AgentNode
from nodes.reranker_node import RerankerNode
from nodes.tool_node import ToolNode
from tools.general_tool import GeneralTool
from tools.rag_tool import RAGTool
from tools.rewrite_tool import RewriteTool


class RAGGraph:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def build(self):
        graph = StateGraph(GraphState)

        # =========================
        # Initialize Nodes
        # =========================
        agent = AgentNode(self.llm)

        tools = {
            "general": GeneralTool(self.llm),
            "rag": RAGTool(self.retriever, RerankerNode(top_k=3), self.llm),
            "rewrite": RewriteTool(self.llm),
        }

        tool_node = ToolNode(tools)

        # =========================
        # Add Nodes to Graph
        # =========================
        graph.add_node("agent", agent)
        graph.add_node("tool", tool_node)

        # =========================
        # Entry Point
        # =========================
        graph.set_entry_point("agent")

        # =========================
        # Flow: Agent → Tool
        # =========================
        graph.add_edge("agent", "tool")

        # =========================
        # Conditional Flow
        # =========================
        graph.add_conditional_edges(
            "tool",
            lambda state: "agent" if state["tool"] == "rewrite" else "end",
            {
                "agent": "agent",  # loop back after rewrite
                "end": END,  # finish after general/rag
            },
        )

        return graph.compile()
