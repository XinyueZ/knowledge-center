import os
import sys
from typing import Any, List, Tuple

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.core import (PromptTemplate, QueryBundle, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

from knowledge_center.completions.vanilla_query_engine import \
    VanillaQueryEngine
from knowledge_center.rags import (default_hyde_embeddings,
                                   default_hyde_gen_llm,
                                   default_hyde_synthesizer_llm,
                                   default_hyde_update_query_llm)
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import lli_from_chroma_store, pretty_print

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)


RERANK_TOP_K = 5


class _HyDERetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, hypo_gen_llm: BaseLLM):
        self.base_retriever = base_retriever
        self.hyde_query_engine = VanillaQueryEngine(llm=hypo_gen_llm)
        self.hypothesis_template = PromptTemplate(
            """Write a hypothetical document about query as you can.

            Only return the document content without any other information, ie. leading text, title text, instructions and so on.
            
            Query: {query}

            """
        )

    def _gen_hypo_doc(self, query_bundle: QueryBundle):
        query_str: str = query_bundle.query_str
        hypo_doc = self.hyde_query_engine(
            self.hypothesis_template.format(query=query_str)
        ).strip()

        return hypo_doc

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pretty_print("input query", query_bundle.query_str)
        hypo_doc = self._gen_hypo_doc(query_bundle)
        pretty_print("hyde result", hypo_doc)
        return self.base_retriever.retrieve(hypo_doc)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pretty_print("input query", query_bundle.query_str)
        hypo_doc = self._gen_hypo_doc(query_bundle)
        pretty_print("hyde result", hypo_doc)
        return await self.base_retriever.aretrieve(hypo_doc)


class HyDE(BaseRAG):

    def __init__(
        self,
        update_query_llm: BaseLLM,
        hypo_gen_llm: BaseLLM,
        synthesizer_llm: BaseLLM,
        embeddings: EmbedType,
    ) -> None:
        self.update_query_llm = update_query_llm
        self.hypo_gen_llm = hypo_gen_llm
        self.synthesizer_llm = synthesizer_llm
        self.embeddings = embeddings

    def _create_base_retriever_and_query_engine(
        self,
        persist_directory: str,
        index_name: str,
    ) -> Tuple[BaseRetriever, BaseQueryEngine]:
        store = lli_from_chroma_store(persist_directory, index_name)
        vector_index: BaseIndex = VectorStoreIndex.from_vector_store(
            store, embed_model=self.embeddings
        )

        retriever = vector_index.as_retriever()
        query_engine = vector_index.as_query_engine(llm=self.update_query_llm)
        return retriever, query_engine

    def _create_hyde_query_engine(
        self,
        base_retriever: BaseRetriever,
    ) -> BaseQueryEngine:
        hyde_retriever = _HyDERetriever(base_retriever, self.hypo_gen_llm)

        rerank: BaseNodePostprocessor = SentenceTransformerRerank(
            top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
        )
        response_synthesizer: BaseSynthesizer = get_response_synthesizer(
            response_mode=ResponseMode.REFINE, llm=self.synthesizer_llm
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
        updated_query = base_query_engine.query(
            prompt.format(origin_query=kwds["query"])
        )

        pretty_print("updated query", updated_query)

        query_engine = self._create_hyde_query_engine(base_retriever=base_retriever)
        return query_engine.query(updated_query.response)


def main():
    hyde = HyDE(
        update_query_llm=default_hyde_update_query_llm,
        hypo_gen_llm=default_hyde_gen_llm,
        synthesizer_llm=default_hyde_synthesizer_llm,
        embeddings=default_hyde_embeddings,
    )

    hyde_res = hyde(
        index_name="4th",
        persist_directory="./vector_db",
        query="These documents outline a company's policies regarding employee conduct and information disclosure. Employees are instructed to direct media and shareholder requests for information to the Investor Relations Manager and to provide written acknowledgement of certain procedures. The company also intends to conduct fair investigations while preserving confidentiality.",
    )

    pretty_print("final res", str(hyde_res))


if __name__ == "__main__":
    main()
