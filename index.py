from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_parser import DocumentParser
from embeddings.hf_embeddings import get_embeddings
from indexing.vector_store import create_or_update_db

# 1. Load documents
parser = DocumentParser()
docs = parser.load("data/")  # your folder path

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
split_docs = splitter.split_documents(docs)

# 3. Embeddings
embeddings = get_embeddings()  # or HuggingFace, etc.

# 4. Create / update DB
create_or_update_db(split_docs, embeddings)

print("Indexing complete")
