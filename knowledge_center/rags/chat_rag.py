import os
import sys
from typing import Any, Dict, List, Sequence, Union

from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.chat_engine import (CondensePlusContextChatEngine,
                                          CondenseQuestionChatEngine,
                                          ContextChatEngine)
from llama_index.core.chat_engine.types import (AGENT_CHAT_RESPONSE_TYPE,
                                                BaseChatEngine,
                                                ChatResponseMode,
                                                StreamingAgentChatResponse)
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices.query.query_transform.base import \
    StepDecomposeQueryTransform
from llama_index.core.query_engine import (BaseQueryEngine,
                                           MultiStepQueryEngine,
                                           RetrieverQueryEngine)
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import (RecursiveRetriever, RouterRetriever,
                                         VectorIndexRetriever)
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, RetrieverTool, ToolMetadata
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.legacy.postprocessor import SentenceTransformerRerank
from llama_index.llms.langchain.base import LangChainLLM

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)


from knowledge_center.completions.vanilla_query_engine import \
    VanillaQueryEngine
from knowledge_center.dashboard.description_crud import (connect_db,
                                                         fetch_descriptions)
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import (N_MULTI_STEPS, RERANK_TOP_K, SIM_TOP_K,
                                    VERBOSE, lli_from_chroma_store,
                                    pretty_print)


class ChatRAG(BaseRAG):
    engine: Union[BaseQueryEngine | BaseChatEngine]
    stream: bool

    def __init__(
        self,
        llm: BaseLLM,
        persist_directory: str,
        streaming: bool = False,
        verbose=VERBOSE,
    ) -> None:
        self.stream = streaming
        index_name_list = [
            name
            for name in os.listdir(persist_directory)
            if os.path.isdir(os.path.join(persist_directory, name))
        ]
        data = fetch_descriptions(conn=connect_db())
        descriptions = list(map(lambda x: x[1], data))
        embeddings_names = list(map(lambda x: x[3], data))
        index_list: List[VectorStoreIndex] = [
            VectorStoreIndex.from_vector_store(
                lli_from_chroma_store(persist_directory, index_name),
                LangchainEmbedding(embeddings_fn_lookup[embeds]()),
            )
            for index_name, embeds in zip(index_name_list, embeddings_names)
        ]
        descriptions = list(map(lambda x: x[1], fetch_descriptions(conn=connect_db())))

        retrievers = [
            RecursiveRetriever(
                "vector",
                retriever_dict={
                    "vector": index.as_retriever(similarity_top_k=SIM_TOP_K)
                },
                verbose=verbose,
            )
            for index in index_list
        ]
        query_engines: List[BaseQueryEngine] = [
            RetrieverQueryEngine.from_args(
                retriever,
                response_synthesizer=get_response_synthesizer(
                    streaming=streaming, llm=llm
                ),
                llm=llm,
                node_postprocessors=[
                    SentenceTransformerRerank(
                        top_n=RERANK_TOP_K, model="BAAI/bge-reranker-base"
                    )
                ],
                verbose=verbose,
            )
            for retriever in retrievers
        ]

        ###### Just query not question ######
        query_engine_tools = [
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=index_name,
                    description="""Useful for queries (not questions) on the content that covers the following dedicated topic:
---Topic:---
{topic}.
---Notice:---
The topic comes from the documents that have been igested in the index in the vector database.
""".format(
                        topic=description
                    ),
                ),
            )
            for index_name, engine, description in zip(
                index_name_list, query_engines, descriptions
            )
        ]
        mix_query_tool = QueryEngineTool(
            query_engine=ReActAgent.from_llm(
                tools=query_engine_tools,
                llm=llm,
                streaming=streaming,
                verbose=verbose,
            ),
            metadata=ToolMetadata(
                name="Mix query tool",
                description="""Useful for the queries (not questions) that cross all the contexts and documents,
don't use the information outside those contexts, just say "I don't know." The topics to context:
---Topics:---
{topics}.
""".format(
                    topics="\n---One Topic---\n".join(descriptions),
                ),
            ),
        )
        ###### Just question ######
        question_engine_tools = [
            QueryEngineTool(
                query_engine=MultiStepQueryEngine(
                    query_engine=engine,
                    query_transform=StepDecomposeQueryTransform(
                        llm=llm, verbose=verbose
                    ),
                    num_steps=N_MULTI_STEPS,
                ),
                metadata=ToolMetadata(
                    name=index_name,
                    description="""Useful for queries (questions) on the content that covers the following dedicated topic:
---Topic:---
{topic}.
---Notice:---
The topic comes from the documents that have been igested in the index in the vector database.
""".format(
                        topic=description
                    ),
                ),
            )
            for index_name, engine, description in zip(
                index_name_list, query_engines, descriptions
            )
        ]
        mix_questions_tool = QueryEngineTool(
            query_engine=ReActAgent.from_llm(
                tools=question_engine_tools,
                llm=llm,
                streaming=streaming,
                verbose=verbose,
            ),
            metadata=ToolMetadata(
                name="Mix query tool",
                description="""Useful for the queries (questions) that cross all the contexts and documents,
don't use the information outside those contexts, just say "I don't know." The topics to context:
---Topics:---
{topics}.
""".format(
                    topics="\n---One Topic---\n".join(descriptions),
                ),
            ),
        )

        ###### Fallback tools ######
        fallback_tools = [
            QueryEngineTool(
                query_engine=VanillaQueryEngine(llm=llm),
                metadata=ToolMetadata(
                    name="General query tool",
                    description="""Useful for the any general queries if cannot answer based on the documents and context.""",
                ),
            )
        ]

        ###### Bind all together ######
        self.engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(
                llm=llm,
                prompt_template_str="Select only the content that is most relevant to the query.",
            ),
            query_engine_tools=query_engine_tools
            + [mix_query_tool]
            + question_engine_tools
            + [mix_questions_tool]
            + fallback_tools,
            summarizer=TreeSummarize(
                streaming=streaming,
                use_async=False,
                verbose=verbose,
            ),
            verbose=verbose,
        )
        self.engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=self.engine,
            llm=llm,
            streaming=streaming,
            verbose=verbose,
        )

    def __call__(
        self, *args: Any, **kwds: Any
    ) -> Union[AGENT_CHAT_RESPONSE_TYPE | RESPONSE_TYPE]:
        message = (
            args[0] if args else kwds["message"] if kwds and "message" in kwds else ""
        )
        res = self.engine.chat(message)
        if not self.stream:
            res.is_dummy_stream = True
        return res


def main():
    chat_rag = ChatRAG(
        llm=LangChainLLM(llms_fn_lookup["Ollama/mistral"]()),
        persist_directory="./vector_db",
        verbose=True,
    )

    pretty_print("I say", "HI")
    res = chat_rag("HI")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Briefing all the content you have.")
    res = chat_rag("Briefing all the content you have.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Briefing the 2nd content you have.")
    res = chat_rag("Briefing the 2nd content you have.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "hey, my name is Bob.")
    res = chat_rag("hey, my name is Bob.")
    pretty_print("AI say", res.response)

    pretty_print("I say", "Repeat my name if you could have remembered.")
    res = chat_rag("Repeat my name if you could have remembered.")
    pretty_print("AI say", res.response)


if __name__ == "__main__":
    main()
