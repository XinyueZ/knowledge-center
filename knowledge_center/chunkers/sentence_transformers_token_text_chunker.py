from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from knowledge_center.chunkers.base_chunker import BaseChunker, ChunkOverlap


class SentenceTransformersTokenTextChunker(BaseChunker, ChunkOverlap):
    def __init__(self, chunk_overlap: int, embeddings: Embeddings) -> None:
        super().__init__(embeddings)
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, documents=Iterable[Document]) -> List[Document]:
        splitter = SentenceTransformersTokenTextSplitter.from_tiktoken_encoder(
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents=documents)
        return chunks
