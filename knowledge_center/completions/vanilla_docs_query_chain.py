from typing import List

from langchain_core.documents import Document

from knowledge_center.rags.vanilla_rag import VanillaRAG


class VanillaDocsQueryChain:
    def __init__(self):
        self.rag = VanillaRAG()

    def __call__(self, documents: List[Document]) -> str:
        query = {
            "documents": documents,
            "prompt": "Description of the documents",
            "preamble": "You're an AI assistant to get the description of the documents briefly.",
        }
        return self.rag(**query)


def main():
    docs = [
        Document("This is a document of Tom"),
        Document("This is another document of Jerry"),
        Document("This is a document of Tom and Jerry"),
    ]
    query_chain = VanillaDocsQueryChain()
    print(query_chain(docs))


if __name__ == "__main__":
    main()
