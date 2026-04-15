import os

from dotenv import load_dotenv
from langsmith import traceable
from tavily import TavilyClient

load_dotenv()


class WebSearchTool:
    """Tool for web search using Tavily API."""

    def __init__(self, llm):
        self.llm = llm
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    @traceable(name="web_search_tool")
    def run(self, query):
        response = self.client.search(
            query=query,
            search_depth="advanced",
        )

        results = [r["content"] for r in response["results"][:5]]
        context = "\n".join(results)

        prompt = (
            "Answer using the web search results.\n"
            "\n"
            f"Context:\n{context}\n"
            "\n"
            f"Question:\n{query}"
        )

        return self.llm.invoke(prompt)
