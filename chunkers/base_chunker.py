import os
from typing import Any, Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class BaseChunker:
    embeddings: Embeddings

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.chunk(**kwds)

    def create_chunks(self, documents: Iterable[Document]) -> List[Document]:
        raise NotImplementedError

    def chunk(
        self,
        documents: Iterable[Document],
        persist_directory: str,
        index_name: str,
    ) -> None:
        chunks = self.create_chunks(documents)
        persist_index_directory = os.path.join(persist_directory, index_name)
        Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=persist_index_directory,
        )


class ChunkOverlap:
    chunk_overlap: int


class ChunkSize:
    chunk_size: int
