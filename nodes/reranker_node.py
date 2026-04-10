from sentence_transformers import CrossEncoder

# model name assigned to variable to avoid long literal line
MODEL_NAME = "BAAI/bge-reranker-base"


class RerankerNode:
    def __init__(self, top_k=3):
        # BAAI reranker model
        self.model = CrossEncoder(MODEL_NAME)
        self.top_k = top_k

    def __call__(self, state):
        query = state["query"]
        contexts = state["context"]

        # Pair query with each chunk
        pairs = [(query, doc) for doc in contexts]

        # Get scores
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            zip(contexts, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Keep top-k
        top_contexts = [doc for doc, _ in ranked[: self.top_k]]

        state["context"] = top_contexts
        return state
