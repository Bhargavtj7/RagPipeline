from typing import List, TypedDict


class GraphState(TypedDict):
    query: str
    context: List[str]
    answer: str
    tool: str
