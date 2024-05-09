import os
import sys
from typing import Any, Dict, List, Union

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import (RetrieverQueryEngine,
                                           SubQuestionQueryEngine)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.postprocessor import SentenceTransformerRerank
from llama_index.llms.langchain.base import LangChainLLM

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)


from knowledge_center.dashboard.description_crud import (
    connect_db, fetch_description_by_index)
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import (RERANK_TOP_K, SIM_TOP_K, VERBOSE,
                                    get_nodes_from_vector_index,
                                    lli_from_chroma_store, pretty_print)


class SubQueriesRAG(BaseRAG):
    engine: BaseQueryEngine
    streaming: bool

    def __init__(
        self,
        llm: BaseLLM,
        persist_directory: str,
        index_name: str,
        streaming: bool = False,
        verbose=VERBOSE,
    ) -> None:
        if not os.path.exists(persist_directory):
            pretty_print("Notice", f"Persist directory {persist_directory} not found.")
            raise FileNotFoundError(f"Persist directory {persist_directory} not found.")
        self.streaming = streaming
        data = fetch_description_by_index(conn=connect_db(), index_name=index_name)
        embeddings_name = data[3]
        index = VectorStoreIndex.from_vector_store(
            lli_from_chroma_store(persist_directory, index_name),
            LangchainEmbedding(embeddings_fn_lookup[embeddings_name]()),
        )
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": index.as_retriever(similarity_top_k=SIM_TOP_K)},
            node_dict=get_nodes_from_vector_index(index),
            verbose=verbose,
        )
        response_synthesizer = get_response_synthesizer(
            streaming=streaming,
            llm=llm,
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=RetrieverQueryEngine.from_args(
                retriever,
                response_synthesizer=response_synthesizer,
                llm=llm,
                node_postprocessors=[
                    SentenceTransformerRerank(
                        top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
                    )
                ],
                streaming=streaming,
                verbose=verbose,
            )
        )
        self.engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[query_engine_tool],
            response_synthesizer=response_synthesizer,
            llm=llm,
            verbose=verbose,
            use_async=True,
        )

    def query(self, message: str) -> RESPONSE_TYPE:
        return self(message)

    async def query_async(self, message: str) -> RESPONSE_TYPE:
        return await self(message)

    def __call__(self, *args: Any, **kwds: Any) -> RESPONSE_TYPE:
        message = (
            args[0] if args else kwds["message"] if kwds and "message" in kwds else ""
        )
        res = self.engine.query(message)
        return res


def main():
    rag = SubQueriesRAG(
        llm=LangChainLLM(llms_fn_lookup["Ollama/mistral"]()),
        index_name="test",
        persist_directory="./vector_db",
        verbose=True,
    )

    pretty_print("I say", "HI")
    res = rag("HI")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Briefing all the content you have.")
    res = rag("Briefing all the content you have.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Briefing the 2nd content you have.")
    res = rag("Briefing the 2nd content you have.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "hey, my name is Bob.")
    res = rag("hey, my name is Bob.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Repeat my name if you could have remembered.")
    res = rag("Repeat my name if you could have remembered.")
    pretty_print("AI say", res.response)


if __name__ == "__main__":
    main()
