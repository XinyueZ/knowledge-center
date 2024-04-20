import os
from typing import Iterable, List, Union

import chromadb
from chromadb.api import ClientAPI
from langchain_core.documents import Document as lc_Document
from langchain_core.embeddings import Embeddings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document as lli_Document
from llama_index.legacy.core.embeddings.base import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from knowledge_center.chunkers.base_chunker import BaseChunker

WIN_SZ = 3


class SentenceWindowChunker(BaseChunker):

    def __init__(self, embeddings: Union[Embeddings, BaseEmbedding]) -> None:
        super().__init__(embeddings)

    @classmethod
    def _cnvt(cls, lc_doc: lc_Document) -> lli_Document:
        lli_doc = lli_Document.from_langchain_format(lc_doc)
        return lli_doc

    def create_chunks(
        self, documents: Union[Iterable[lc_Document], Iterable[lli_Document]]
    ) -> Union[List[lc_Document], List[BaseNode]]:
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=WIN_SZ,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        docs = list(map(SentenceWindowChunker._cnvt, documents))
        nodes: List[BaseNode] = node_parser.build_window_nodes_from_documents(docs)

        return nodes

    def chunk(
        self,
        documents: Union[Iterable[lc_Document], Iterable[lli_Document]],
        persist_directory: str,
        index_name: str,
    ) -> None:
        chunks = self.create_chunks(documents)

        path = os.path.join(persist_directory, index_name)
        db: ClientAPI = chromadb.PersistentClient(path=path)
        chroma_collection = db.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(
            chunks, storage_context=storage_context, embed_model=self.embeddings
        )
