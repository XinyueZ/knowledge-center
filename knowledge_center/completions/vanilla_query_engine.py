import os
import sys
from typing import Any

from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.groq import Groq

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from knowledge_center.completions.base_completion import BaseCompletion
from knowledge_center.utils import pretty_print

default_vanilla_llm = Groq(
    model="llama3-70b-8192",  # llama3-8b-8192
    temperature=0,
    timeout=60,
)


class VanillaQueryEngine(CustomQueryEngine, BaseCompletion):
    """RAG String Query Engine."""

    llm: LLM

    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.custom_query(*args, **kwds)

    def custom_query(self, query_str: str) -> str:
        return str(self.llm.complete(query_str))


def main():
    vanilla_query_engine = VanillaQueryEngine(llm=default_vanilla_llm)
    query_res = vanilla_query_engine(query_str="What is the capital of France?")
    pretty_print("query_res", query_res)


if __name__ == "__main__":
    main()
