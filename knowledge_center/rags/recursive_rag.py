import asyncio
import os
import sys
from math import e
from typing import Any, Sequence

from llama_index.core import (SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, get_response_synthesizer,
                              load_index_from_storage)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor.metadata_replacement import \
    MetadataReplacementPostProcessor
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.postprocessor import SentenceTransformerRerank
from llama_index.llms.langchain.base import LangChainLLM

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from llama_index.core.embeddings.utils import EmbedType

from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import RERANK_TOP_K, SIM_TOP_K, VERBOSE, WIN_SZ


class RecursiveRAG(BaseRAG):

    query_engine: BaseQueryEngine

    def __init__(
        self,
        llm: LLM,
        embeddings: EmbedType,
        docs: Sequence[Document],
        persist_directory: str,
        streaming=True,
        verbose=VERBOSE,
    ) -> None:
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=WIN_SZ,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        synth = get_response_synthesizer(streaming=streaming, llm=llm)
        nodes = node_parser.build_window_nodes_from_documents(docs)
        if os.path.exists(persist_directory):
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(
                    persist_dir=persist_directory
                ),
                vector_store=SimpleVectorStore.from_persist_path(
                    persist_path=os.path.join(
                        persist_directory, "default__vector_store.json"
                    )
                ),
                index_store=SimpleIndexStore.from_persist_dir(
                    persist_dir=persist_directory
                ),
            )
            index = load_index_from_storage(storage_context, embed_model=embeddings)
        else:
            index = VectorStoreIndex(
                nodes,
                embed_model=embeddings,
                show_progress=True,
            )
            index.storage_context.persist(persist_dir=persist_directory)
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": index.as_retriever(similarity_top_k=SIM_TOP_K)},
            verbose=verbose,
        )
        rerank = SentenceTransformerRerank(
            top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever,
            response_synthesizer=synth,
            llm=llm,
            node_postprocessors=[postproc, rerank],
            verbose=verbose,
            streaming=streaming,
        )

    async def aquery(self, query: str) -> RESPONSE_TYPE:
        return self.query(query=query)

    def query(self, query: str) -> RESPONSE_TYPE:
        return self(query=query)

    def __call__(self, *args: Any, **kwds: Any) -> RESPONSE_TYPE:
        return self.query_engine.query(kwds["query"])


async def main():
    docs = SimpleDirectoryReader(input_files=["README.md"]).load_data()
    rag = RecursiveRAG(
        llm=LangChainLLM(llms_fn_lookup["Ollama/mistral"]()),
        persist_directory="./knowledge_center/chat/vector_db",
        embeddings=LangchainEmbedding(
            embeddings_fn_lookup["Ollama/nomic-embed-text"]()
        ),
        docs=docs,
    )
    query_res = await rag.aquery(
        query="Briefing the description of the documents as context for the query in detail."
    )
    query_res.print_response_stream()


if __name__ == "__main__":
    asyncio.run(main())
