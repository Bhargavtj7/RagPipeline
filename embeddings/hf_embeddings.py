from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """Get HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
