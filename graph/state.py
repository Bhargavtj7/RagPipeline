from typing import TypedDict


class GraphState(TypedDict):
    """State schema for the RAG graph."""

    query: str
    context: list[str]
    answer: str
    tool: str
