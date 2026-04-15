import os

from langchain_community.vectorstores import Chroma

import utils.langsmith_config  # noqa: F401
from document_parser import DocumentParser
from embeddings.hf_embeddings import get_embeddings
from graph.rag_graph import RAGGraph
from llm.groq_llm import GroqLLM
from utils.text_splitter import split_documents
from vectorstore.chroma_store import create_vector_store

DB_PATH = "./chroma_db"


def load_db(embeddings):
    """Load existing Chroma database."""
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )


def main():
    """Main RAG pipeline execution."""
    # 1. Load embeddings
    embeddings = get_embeddings()

    # 2. Check if DB exists
    if os.path.exists(DB_PATH):
        print("Loading existing Vector DB...")
        vectordb = load_db(embeddings)

    else:
        print("No DB found. Creating new one...")

        # Parse Documents
        parser = DocumentParser()
        docs = parser.parse_directory("./data")

        print("Total docs:", len(docs))

        if not docs:
            print("No documents found.")
            return

        # Split by file type
        pdf_docs = [
            doc for doc in docs if doc.metadata.get("source", "").endswith(".pdf")
        ]
        csv_docs = [
            doc for doc in docs if doc.metadata.get("source", "").endswith(".csv")
        ]
        html_docs = [
            doc for doc in docs if doc.metadata.get("source", "").endswith(".html")
        ]

        # Split documents
        split_pdf_docs = split_documents(pdf_docs)
        split_csv_docs = split_documents(csv_docs)
        split_html_docs = split_documents(html_docs)

        # Create vector stores (same DB, different collections)
        create_vector_store(split_pdf_docs, embeddings, "pdf_docs")
        create_vector_store(split_csv_docs, embeddings, "csv_data")
        create_vector_store(split_html_docs, embeddings, "html_docs")

        print("Vector DBs created successfully!")

        # Load DB after creation
        vectordb = load_db(embeddings)

    # 3. Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # 4. Initialize LLM + Graph
    llm = GroqLLM()
    app = RAGGraph(llm, retriever).build()

    # 5. Query Loop
    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        result = app.invoke(
            {
                "query": query,
                "context": [],
                "answer": "",
                "tool": "",
            }
        )

        print("\nAnswer:\n", result["answer"])


if __name__ == "__main__":
    main()
