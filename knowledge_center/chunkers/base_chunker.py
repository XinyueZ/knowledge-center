import os
from typing import Any, Iterable, List, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document as lc_Document
from langchain_core.embeddings import Embeddings
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document as lli_Document
from llama_index.legacy.core.embeddings.base import BaseEmbedding


class BaseChunker:
    embeddings: Union[Embeddings, BaseEmbedding]

    def __init__(self, embeddings: Union[Embeddings, BaseEmbedding]) -> None:
        self.embeddings = embeddings

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.chunk(**kwds)

    def create_chunks(
        self, documents: Union[Iterable[lc_Document], Iterable[lli_Document]]
    ) -> Union[List[lc_Document], List[BaseNode]]:
        raise NotImplementedError

    def chunk(
        self,
        documents: Union[Iterable[lc_Document], Iterable[lli_Document]],
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
