from document_parser import DocumentParser
from embeddings.hf_embeddings import get_embeddings
from graph.rag_graph import RAGGraph
from llm.groq_llm import GroqLLM
from retriever.merger_retriever import CombinedRetriever
from utils.text_splitter import split_documents
from vectorstore.chroma_store import create_vector_store


def main():
    # =========================
    # 1. Parse Documents
    # =========================
    parser = DocumentParser()
    docs = parser.parse_directory("./data")

    print("Total docs:", len(docs))

    if not docs:
        print("No documents found.")
        return

    # =========================
    # 2. Split by file type
    # =========================
    pdf_docs = [doc for doc in docs if doc.metadata.get("source", "").endswith(".pdf")]

    csv_docs = [doc for doc in docs if doc.metadata.get("source", "").endswith(".csv")]

    html_docs = [
        doc for doc in docs if doc.metadata.get("source", "").endswith(".html")
    ]
    # =========================
    # 3. Split documents
    # =========================
    split_pdf_docs = split_documents(pdf_docs)
    split_csv_docs = split_documents(csv_docs)
    split_html_docs = split_documents(html_docs)

    # =========================
    # 4. Load embeddings
    # =========================
    embeddings = get_embeddings()

    # =========================
    # 5. Create vector stores
    # =========================
    pdf_store = create_vector_store(
        split_pdf_docs,
        embeddings,
        "pdf_docs",
    )

    csv_store = create_vector_store(
        split_csv_docs,
        embeddings,
        "csv_data",
    )

    html_store = create_vector_store(
        split_html_docs,
        embeddings,
        "html_docs",
    )

    print("✅ Vector DBs created successfully!")

    # =========================
    # 6. Create retrievers
    # =========================
    pdf_retriever = pdf_store.as_retriever(search_kwargs={"k": 5})
    csv_retriever = csv_store.as_retriever(search_kwargs={"k": 5})
    html_retriever = html_store.as_retriever(search_kwargs={"k": 5})

    # ✅ Combined Retriever
    retriever = CombinedRetriever(
        [
            pdf_retriever,
            csv_retriever,
            html_retriever,
        ]
    )

    # =========================
    # 7. Initialize LLM + Graph
    # =========================
    llm = GroqLLM()
    app = RAGGraph(llm, retriever).build()

    # =========================
    # 8. Interactive Query Loop
    # =========================
    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        result = app.invoke(
            {
                "query": query,
                "context": [],
                "is_general": False,
                "answer": "",
            }
        )

        print("\n💡 Answer:\n", result["answer"])


if __name__ == "__main__":
    main()
