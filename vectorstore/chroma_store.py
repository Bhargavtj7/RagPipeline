from langchain_community.vectorstores import Chroma


def create_vector_store(
    documents,
    embeddings,
    collection_name="default_collection",
):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name,
    )

    vectorstore.persist()
    return vectorstore
