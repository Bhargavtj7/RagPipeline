class GeneralTool:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query):
        return self.llm.invoke(query)
