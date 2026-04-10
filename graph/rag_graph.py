from langgraph.graph import END, StateGraph

from graph.state import GraphState
from nodes.answer_node import AnswerNode
from nodes.general_node import GeneralNode
from nodes.reranker_node import RerankerNode
from nodes.retriever_node import RetrieverNode
from nodes.router_node import RouterNode


class RAGGraph:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def build(self):
        graph = StateGraph(GraphState)

        # Nodes
        router = RouterNode(self.llm)
        general = GeneralNode(self.llm)
        retriever = RetrieverNode(self.retriever)
        reranker = RerankerNode(top_k=3)  # ✅ NEW
        answer = AnswerNode(self.llm)

        graph.add_node("router", router)
        graph.add_node("general", general)
        graph.add_node("retriever", retriever)
        graph.add_node("reranker", reranker)  # ✅ NEW
        graph.add_node("answer", answer)

        # Entry
        graph.set_entry_point("router")

        # Router logic
        graph.add_conditional_edges(
            "router",
            lambda state: "general" if state["is_general"] else "rag",
            {"general": "general", "rag": "retriever"},
        )

        # ✅ Updated flow (NO validator)
        graph.add_edge("retriever", "reranker")
        graph.add_edge("reranker", "answer")

        # End nodes
        graph.add_edge("general", END)
        graph.add_edge("answer", END)

        return graph.compile()
