from typing import List, TypedDict


class GraphState(TypedDict):
    query: str
    context: List[str]
    is_general: bool
    is_relevant: bool
    answer: str
