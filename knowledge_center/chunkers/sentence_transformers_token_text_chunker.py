from typing import Iterable, List, Union

from langchain_core.documents import Document as lc_Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document as lli_Document
from llama_index.legacy.core.embeddings.base import BaseEmbedding

from knowledge_center.chunkers.base_chunker import BaseChunker, ChunkOverlap


class SentenceTransformersTokenTextChunker(BaseChunker, ChunkOverlap):
    def __init__(
        self, chunk_overlap: int, embeddings: Union[Embeddings, BaseEmbedding]
    ) -> None:
        super().__init__(embeddings)
        self.chunk_overlap = chunk_overlap

    def create_chunks(
        self, documents=Union[Iterable[lc_Document], Iterable[lli_Document]]
    ) -> Union[List[lc_Document], List[BaseNode]]:
        splitter = SentenceTransformersTokenTextSplitter.from_tiktoken_encoder(
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents=documents)
        return chunks
