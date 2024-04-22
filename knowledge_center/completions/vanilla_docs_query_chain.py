import os
import sys
from typing import List

from langchain_core.documents import Document

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from typing import Any

from langchain_core.language_models.base import BaseLanguageModel

from knowledge_center.completions.base_completion import BaseCompletion
from knowledge_center.models.llms import llms_lookup
from knowledge_center.rags.vanilla_rag import VanillaRAG
from knowledge_center.utils import pretty_print


class VanillaDocsQueryChain(BaseCompletion):
    def __init__(self, llm: BaseLanguageModel):
        self.rag = VanillaRAG(llm)

    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.query(*args, **kwds)

    def query(self, documents: List[Document], preamble: str) -> str:
        query = {
            "documents": documents,
            "prompt": "Briefing the description of the documents as context.",
            "preamble": preamble,
        }
        return self.rag(**query)


def main():
    docs = [
        Document("This is a document of Tom"),
        Document("This is another document of Jerry"),
        Document("This is a document of Tom and Jerry"),
    ]
    query_chain = VanillaDocsQueryChain(llm=llms_lookup["ChatCohere"]())
    query_res = query_chain(
        documents=docs,
        preamble="You're an AI assistant to get the description of the documents briefly.",
    )

    pretty_print("query_res", query_res)


if __name__ == "__main__":
    main()
