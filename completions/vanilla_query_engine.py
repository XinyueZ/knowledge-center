from llama_index.llms.groq import Groq
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import CustomQueryEngine

default_vanilla_llm = Groq(
    model="llama3-70b-8192",  # llama3-8b-8192
    temperature=0,
    timeout=60,
)


class VanillaQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    llm: LLM

    def custom_query(self, query_str: str) -> str:
        return str(self.llm.complete(query_str))


def main():
    vanilla_query_engine = VanillaQueryEngine(llm=default_vanilla_llm)
    res_str = vanilla_query_engine.custom_query("What is the capital of France?")
    print(res_str)


if __name__ == "__main__":
    main()
