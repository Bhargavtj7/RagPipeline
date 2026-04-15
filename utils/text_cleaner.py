import re


def clean_text(text: str) -> str:
    """Clean text by removing unwanted patterns."""
    # Remove image markdown
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove filenames like _page_7_Figure_7.jpeg
    text = re.sub(r"_page_\d+_Figure_\d+\.\w+", "", text)

    # Remove repeated "Result X:" lines
    text = re.sub(r"Result \d+:", "", text)

    # Remove horizontal rules / multiple dashes
    text = re.sub(r"-{3,}", "", text)

    # Remove extra spaces / newlines
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
