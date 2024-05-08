import os
import sys
from typing import Any, List, Tuple

from llama_index.core import (PromptTemplate, QueryBundle, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever, RecursiveRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain.base import LangChainLLM

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import (RERANK_TOP_K, SIM_TOP_K, VERBOSE,
                                    get_nodes_from_vector_index,
                                    lli_from_chroma_store, pretty_print)


class _HyDERetriever(BaseRetriever):
    def __init__(
        self, base_retriever: BaseRetriever, base_query_engine: BaseQueryEngine
    ):  # , hypo_gen_llm: BaseLLM):
        self.base_retriever = base_retriever
        self.base_query_engine = base_query_engine
        self.hypothesis_template = PromptTemplate(
            """Give a hypothetical paper about the context inside >>> and <<< marks.
ONLY return the paper content as response without any other information, ie. leading text, title text, instructions and so on.
>>>
{query}
<<<"""
        )

    def _gen_hypo_doc(self, query_bundle: QueryBundle) -> str:
        query_str: str = query_bundle.query_str
        hypo_doc = self.base_query_engine.query(
            self.hypothesis_template.format(query=query_str)
        ).response.strip()

        return hypo_doc

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        hypo_doc = self._gen_hypo_doc(query_bundle)
        pretty_print("HyDE", hypo_doc)
        return self.base_retriever.retrieve(hypo_doc)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        hypo_doc = self._gen_hypo_doc(query_bundle)
        pretty_print("HyDE", hypo_doc)
        return await self.base_retriever.aretrieve(hypo_doc)


class HyDE(BaseRAG):

    def __init__(
        self,
        llm: BaseLLM,
        embeddings: EmbedType,
        streaming: bool = False,
        verbose: bool = VERBOSE,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.streaming = streaming
        self.verbose = verbose

    def _create_base_retriever_and_query_engine(
        self,
        persist_directory: str,
        index_name: str,
    ) -> Tuple[BaseRetriever, BaseQueryEngine]:
        store = lli_from_chroma_store(persist_directory, index_name)
        vector_index = VectorStoreIndex.from_vector_store(
            store, embed_model=self.embeddings
        )
        nodes_dict = get_nodes_from_vector_index(vector_index)
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={
                "vector": vector_index.as_retriever(similarity_top_k=SIM_TOP_K)
            },
            node_dict=nodes_dict,
            verbose=self.verbose,
        )
        query_engine = RetrieverQueryEngine.from_args(
            retriever, llm=self.llm, streaming=self.streaming
        )
        return retriever, query_engine

    def _create_hyde_query_engine(
        self,
        base_retriever: BaseRetriever,
        base_query_engine: BaseQueryEngine,
    ) -> BaseQueryEngine:
        hyde_retriever = _HyDERetriever(base_retriever, base_query_engine)

        rerank: BaseNodePostprocessor = SentenceTransformerRerank(
            top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
        )
        response_synthesizer: BaseSynthesizer = get_response_synthesizer(
            response_mode=ResponseMode.REFINE, llm=self.llm, streaming=self.streaming
        )
        return RetrieverQueryEngine(
            hyde_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[rerank],
        )

    def __call__(self, *args: Any, **kwds: Any) -> RESPONSE_TYPE:
        pretty_print("origin query", kwds["query"])
        base_retriever, base_query_engine = (
            self._create_base_retriever_and_query_engine(
                persist_directory=kwds["persist_directory"],
                index_name=kwds["index_name"],
            )
        )
        prompt = """Generate a different version of query for better retrieval of relevant documents from a vector database.
                    The version is for better model comprehension while maintaining the original text sentiment and brevity.
                    Your goal is to help the user overcome some of the limitations of the distance-based similarity search.
                    Notice: Only return the new version without any explaination, instructions or additional information.

                    Origin query:
                    {origin_query}

                    New query:
                    """
        adjusted_query = base_query_engine.query(
            prompt.format(origin_query=kwds["query"])
        )

        pretty_print("adjusted query", adjusted_query.response)

        query_engine = self._create_hyde_query_engine(
            base_retriever=base_retriever, base_query_engine=base_query_engine
        )
        return query_engine.query(adjusted_query.response)


def main():
    hyde = HyDE(
        llm=LangChainLLM(llms_fn_lookup["Groq/gemma-7b-it"]()),
        embeddings=LangchainEmbedding(embeddings_fn_lookup["NVIDIAEmbeddings"]()),
    )

    hyde_res = hyde(
        index_name="4th",
        persist_directory="./vector_db",
        query="Outline of the document",
    )

    pretty_print("final res", str(hyde_res))


if __name__ == "__main__":
    main()
