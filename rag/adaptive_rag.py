import os
from dataclasses import dataclass
from typing import Any, List

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.core import (Settings, SimpleDirectoryReader,
                              VectorStoreIndex, get_response_synthesizer)
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
from llama_index.core.indices.query.query_transform.base import \
    StepDecomposeQueryTransform
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor.metadata_replacement import \
    MetadataReplacementPostProcessor
from llama_index.core.query_engine import (BaseQueryEngine, CustomQueryEngine,
                                           MultiStepQueryEngine,
                                           RetrieverQueryEngine)
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import Document
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.postprocessor import (CohereRerank,
                                              SentenceTransformerRerank)
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from tqdm.asyncio import tqdm

from rag.base_rag import BaseRAG
from utils import VERBOSE, pretty_print

WIN_SZ = 3
SIM_TOP_K = 5
RERANK_TOP_K = 3
N_MULTI_STEPS = 5


class LLMQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    llm: LLM

    def custom_query(self, query_str: str):
        return str(self.llm.complete(query_str))


@dataclass
class AdaptiveRAGDataSource:
    name: str
    description: str
    query_engine: BaseQueryEngine
    multi_step_query_engine: BaseQueryEngine

    def __hash__(self):
        return hash((self.name, self.description))

    def __eq__(self, other):
        if isinstance(other, AdaptiveRAGDataSource):
            return self.name == other.name and self.description == other.description
        return False


class AdaptiveRAG(BaseRAG):
    summary_llm = Cohere(temperature=0, max_tokens=2048 * 2)
    multi_step_query_engine_llm = Groq(
        model="mixtral-8x7b-32768", temperature=0, timeout=60
    )
    standalone_query_engine_llm = Groq(
        model="mixtral-8x7b-32768", temperature=0, timeout=60
    )
    agent_llm = Anthropic(temperature=0, model="claude-3-haiku-20240307")
    chain_llm = Anthropic(temperature=0, model="claude-3-haiku-20240307")
    general_llm = Anthropic(temperature=0, model="claude-3-haiku-20240307")

    index_dir: str

    def __init__(self, index_dir) -> None:
        Settings.llm = OpenAI(temperature=0, model="gpt-4-turbo")
        Settings.embed_model = LangchainEmbedding(
            NVIDIAEmbeddings(model="nvolveqa_40k")
        )
        # string = "helloworld"
        # temp_res =  Settings.embed_model.get_text_embedding(string)
        # pretty_print("temp_res shape", len(temp_res))

        Settings.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=WIN_SZ,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        self.index_dir = index_dir

    async def load_docs(self, file_paths: List[str]) -> List[AdaptiveRAGDataSource]:
        all_doc_src = []
        tasks = [
            self.index_and_chunks(
                os.path.basename(file_path).split(".")[0],
                SimpleDirectoryReader(input_files=[file_path]).load_data(),
            )
            for file_path in file_paths
        ]
        doc_src_tasks_run = await tqdm.gather(*tasks)
        all_doc_src.extend(doc_src_tasks_run)
        return all_doc_src

    async def index_and_chunks(
        self, file_name: str, raw_docs: List[Document]
    ) -> AdaptiveRAGDataSource:
        pretty_print("Raw docs", file_name)
        name = file_name
        # check if the name is based on String should match pattern '^[a-zA-Z0-9_-]{1,64}$'
        # required by the LlamaIndex.
        # if not, then replace with a valid name
        if not name.isalnum():
            # replace with a valid name
            name = "file_" + str(hash(name))

        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
        )

        # vector indexing
        import faiss
        from llama_index.core import StorageContext
        from llama_index.vector_stores.faiss import FaissVectorStore

        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1024))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            raw_docs, storage_context=storage_context, show_progress=True
        )
        index.storage_context.persist(self.index_dir)
        retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": index.as_retriever(similarity_top_k=SIM_TOP_K)},
            verbose=VERBOSE,
        )

        # summary
        summary = await RetrieverQueryEngine.from_args(
            DocumentSummaryIndex.from_documents(
                raw_docs,
                show_progress=True,
            ).as_retriever(),
            llm=self.summary_llm,
            response_synthesizer=get_response_synthesizer(
                response_mode=ResponseMode.SIMPLE_SUMMARIZE
            ),
            node_postprocessors=[postproc, rerank],
            verbose=VERBOSE,
        ).aquery("Provide the shortest description of the content.")

        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            llm=self.standalone_query_engine_llm,
            node_postprocessors=[postproc, rerank],
            verbose=VERBOSE,
        )
        return AdaptiveRAGDataSource(
            name=name,
            description=summary.response,
            query_engine=query_engine,
            multi_step_query_engine=MultiStepQueryEngine(
                query_engine=query_engine,
                query_transform=StepDecomposeQueryTransform(
                    llm=self.multi_step_query_engine_llm, verbose=VERBOSE
                ),
                num_steps=N_MULTI_STEPS,
            ),
        )

    def build_mulit_step_query_engine_tools(
        self,
        ds_list: List[AdaptiveRAGDataSource],
    ) -> List[QueryEngineTool]:
        desc_fmt = "Useful for complex queries on the content with multi-step that covers the following dedicated topic:\n{topic}\n"
        return [
            QueryEngineTool(
                query_engine=ds.multi_step_query_engine,
                metadata=ToolMetadata(
                    name=ds.name, description=desc_fmt.format(topic=ds.description)
                ),
            )
            for ds in ds_list
        ]

    def build_standalone_query_engine_tools(
        self,
        ds_list: List[AdaptiveRAGDataSource],
    ) -> List[QueryEngineTool]:
        desc_fmt = "Useful for simple queries on the content that covers the following dedicated topic:\n{topic}\n"

        return [
            QueryEngineTool(
                query_engine=ds.query_engine,
                metadata=ToolMetadata(
                    name=ds.name, description=desc_fmt.format(topic=ds.description)
                ),
            )
            for ds in ds_list
        ]

    def build_query_engine_tools_agent_tool(
        self,
        query_engine_tools: List[QueryEngineTool],
        base_description: str,
    ) -> QueryEngineTool:
        agent_worker = FunctionCallingAgentWorker.from_tools(
            query_engine_tools,
            llm=self.agent_llm,
            verbose=VERBOSE,
            allow_parallel_tool_calls=True,
        )
        agent_runner = AgentRunner(
            agent_worker,
            llm=self.agent_llm,
            verbose=VERBOSE,
        )

        description_list = [base_description]
        for tools in query_engine_tools:
            meta = tools.metadata
            description_list.append(
                f"Description of {meta.name}:\n{meta.description}\n"
            )
        description = "\n\n".join(description_list)
        return QueryEngineTool(
            query_engine=agent_runner,
            metadata=ToolMetadata(description=description),
        )

    def build_fallback_query_engine_tool(self) -> QueryEngineTool:
        return QueryEngineTool(
            query_engine=LLMQueryEngine(llm=self.general_llm),
            metadata=ToolMetadata(
                name="General queries as fallback",
                description=(
                    "Useful for information about general queries other than specific data sources, as fallback action if no other tool is selected."
                ),
            ),
        )

    def build_adaptive_rag_chain(
        self, ds_list: List[AdaptiveRAGDataSource]
    ) -> RouterQueryEngine:
        standalone_query_engine_tools = self.build_standalone_query_engine_tools(
            ds_list
        )
        standalone_query_engine_tools_agent_tool = self.build_query_engine_tools_agent_tool(
            self.build_standalone_query_engine_tools(ds_list),
            "Useful for queries that span multiple and cross-docs, the docs should cover different topics:\n",
        )

        multi_step_query_engine_tools = self.build_mulit_step_query_engine_tools(
            ds_list
        )
        multi_step_query_engine_tools_agent_tool = self.build_query_engine_tools_agent_tool(
            self.build_mulit_step_query_engine_tools(ds_list),
            "Useful for complex queries that span multiple and cross-docs with the help of multi-step, the docs should cover different topics:\n",
        )

        fallback_query_engine_tool = self.build_fallback_query_engine_tool()
        query_engine_tools = (
            multi_step_query_engine_tools
            + [multi_step_query_engine_tools_agent_tool]
            + standalone_query_engine_tools
            + [standalone_query_engine_tools_agent_tool]
            + [fallback_query_engine_tool]
        )
        return RouterQueryEngine.from_defaults(
            llm=self.chain_llm,
            selector=LLMSingleSelector.from_defaults(llm=self.chain_llm),
            query_engine_tools=query_engine_tools,
            verbose=VERBOSE,
        )

    async def __call__(self, *args: Any, **kwds: Any) -> RESPONSE_TYPE:
        chain = self.build_adaptive_rag_chain(kwds["ds_list"])
        return chain.query(kwds["query"])
