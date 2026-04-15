from langchain_community.vectorstores import Chroma

from embeddings.hf_embeddings import get_embeddings


def get_retriever(collection_name="documents", k=5):
    """Get retriever for a specific collection."""
    embeddings = get_embeddings()
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever
