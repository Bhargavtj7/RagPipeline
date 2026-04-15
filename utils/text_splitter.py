from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.text_cleaner import clean_text


def split_documents(documents):
    """Split documents into chunks with overlap."""
    # Clean before splitting
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    return splitter.split_documents(documents)
