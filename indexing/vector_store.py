import hashlib
import os

from langchain_community.vectorstores import Chroma

DB_PATH = "./chroma_db"


def get_doc_id(text):
    """Generate unique ID for document based on content hash."""
    return hashlib.md5(text.encode()).hexdigest()


def create_or_update_db(split_docs, embeddings):
    """Create or update vector database with documents."""
    if not os.path.exists(DB_PATH):
        print("Creating new DB...")
        ids = [get_doc_id(doc.page_content) for doc in split_docs]

        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            ids=ids,
            persist_directory=DB_PATH,
        )
    else:
        print("Updating existing DB...")
        vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )

        ids = [get_doc_id(doc.page_content) for doc in split_docs]

        vectordb.add_documents(documents=split_docs, ids=ids)

    return vectordb
