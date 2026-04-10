from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},  # change to "cuda" if GPU
        encode_kwargs={"normalize_embeddings": True},
    )
