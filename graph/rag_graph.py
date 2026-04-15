from langgraph.graph import END, StateGraph

from graph.state import GraphState
from nodes.agent_node import AgentNode
from nodes.reranker_node import RerankerNode
from nodes.tool_node import ToolNode
from tools.general_tool import GeneralTool
from tools.rag_tool import RAGTool
from tools.rewrite_tool import RewriteTool
from tools.web_search_tool import WebSearchTool


class RAGGraph:
    """LangGraph-based RAG pipeline with tool routing."""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def build(self):
        """Build and compile the RAG graph."""
        graph = StateGraph(GraphState)

        agent = AgentNode(self.llm)
        web_tool = WebSearchTool(self.llm)

        tools = {
            "general": GeneralTool(self.llm),
            "rag": RAGTool(
                self.retriever,
                RerankerNode(top_k=3),
                self.llm,
                web_tool,
            ),
            "rewrite": RewriteTool(self.llm),
            "web": web_tool,
        }

        tool_node = ToolNode(tools)

        graph.add_node("agent", agent)
        graph.add_node("tool", tool_node)

        graph.set_entry_point("agent")
        graph.add_edge("agent", "tool")

        graph.add_conditional_edges(
            "tool",
            lambda state: "agent" if state["tool"] == "rewrite" else "end",
            {
                "agent": "agent",
                "end": END,
            },
        )

        return graph.compile()
