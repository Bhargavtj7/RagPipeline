import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load env variables
load_dotenv()


class GroqLLM:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b",
            temperature=0,
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt).content
