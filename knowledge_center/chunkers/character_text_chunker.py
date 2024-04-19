from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter

from knowledge_center.chunkers.base_chunker import (BaseChunker, ChunkOverlap,
                                                    ChunkSize)


class CharacterTextChunker(BaseChunker, ChunkSize, ChunkOverlap):
    def __init__(
        self, chunk_size: int, chunk_overlap: int, embeddings: Embeddings
    ) -> None:
        super().__init__(embeddings)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, documents=Iterable[Document]) -> List[Document]:
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents=documents)
        return chunks
