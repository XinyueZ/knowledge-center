from typing import Iterable, List, Union

from langchain_core.documents import Document as lc_Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document as lli_Document
from llama_index.legacy.core.embeddings.base import BaseEmbedding

from knowledge_center.chunkers.base_chunker import (BaseChunker, ChunkOverlap,
                                                    ChunkSize)


class RecursiveCharacterTextChunker(BaseChunker, ChunkSize, ChunkOverlap):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embeddings: Union[Embeddings, BaseEmbedding],
    ) -> None:
        super().__init__(embeddings)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(
        self, documents=Union[Iterable[lc_Document], Iterable[lli_Document]]
    ) -> Union[List[lc_Document], List[BaseNode]]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents=documents)
        return chunks
