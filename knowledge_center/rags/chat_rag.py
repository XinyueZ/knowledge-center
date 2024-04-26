import os
import sys
from typing import Any, List, Union

from llama_index.core import VectorStoreIndex
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
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, RetrieverTool, ToolMetadata
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
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
from knowledge_center.utils import VERBOSE, lli_from_chroma_store, pretty_print


class ChatRAG(BaseRAG):
    engine: Union[BaseQueryEngine | BaseChatEngine]
    stream: bool

    def __init__(
        self,
        llm: BaseLLM,
        embeddings: EmbedType,
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
        index_list: List[VectorStoreIndex] = [
            VectorStoreIndex.from_vector_store(
                lli_from_chroma_store(persist_directory, index_name), embeddings
            )
            for index_name in index_name_list
        ]
        descriptions = fetch_descriptions(conn=connect_db())
        desc_fmt = """Useful for queries on the content that covers the following dedicated topic:
---Topic:---
{topic}.
---Notice:---
The topic comes from the documents that have been igested in the index in the vector database.
"""

        #         retrievers: List[BaseQueryEngine] = [
        #             index.as_retriever() for index in index_list
        #         ]
        #         retriever_tools = [
        #             RetrieverTool(
        #                 retriever=retriever,
        #                 metadata=ToolMetadata(
        #                     name=index_name, description=desc_fmt.format(topic=description)
        #                 ),
        #             )
        #             for index_name, retriever, description in zip(
        #                 index_name_list, retrievers, descriptions
        #             )
        #         ]
        #         retriever = RouterRetriever(
        #             selector=LLMSingleSelector.from_defaults(llm=llm),
        #             retriever_tools=retriever_tools,
        #         )
        #         self.engine = CondensePlusContextChatEngine.from_defaults(
        #             retriever=retriever,
        #             llm=llm,
        #             system_prompt="""Answer the questions that user asked,
        # if you don't know the answer based on the documents, just say "I don't know." or simple greeting, key the answer
        # as simple as possible, without any additional information, instructions, or examples.
        # """,
        #         )

        engines: List[BaseQueryEngine] = [
            index.as_query_engine(llm=llm, streaming=True) for index in index_list
        ]
        query_engine_tools = [
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=index_name, description=desc_fmt.format(topic=description)
                ),
            )
            for index_name, engine, description in zip(
                index_name_list, engines, descriptions
            )
        ]
        # query_engine_tools += [
        #     QueryEngineTool(
        #         query_engine=VanillaQueryEngine(llm=llm),
        #         metadata=ToolMetadata(
        #             name="General query tool",
        #             description="""Useful for the general queries if cannot answer based on the documents and context.""",
        #         ),
        #     )
        # ]
        mix_tool = QueryEngineTool(
            query_engine=ReActAgent.from_llm(
                tools=query_engine_tools,
                llm=llm,
                streaming=streaming,
                verbose=verbose,
            ),
            metadata=ToolMetadata(
                name="Mix query tool",
                description="""Useful for the queries that cross all the contexts or documents, 
don't use the information outside those contexts, just say "I don't know." """,
            ),
        )

        self.engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=llm),
            query_engine_tools=query_engine_tools + [mix_tool],
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

        # self.engine = ReActAgent.from_llm(
        #     tools=retriever_tools, #query_engine_tools
        #     llm=llm,
        #     streaming=streaming,
        #     verbose=verbose,
        # )

        # self.engine = CondenseQuestionChatEngine.from_defaults(
        #     query_engine=self.engine,
        #     llm=llm,
        #     streaming=streaming,
        #     verbose=verbose,
        # )

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
        llm=LangChainLLM(llms_fn_lookup["Ollama/command-r"]()),
        embeddings=LangchainEmbedding(
            embeddings_fn_lookup["Ollama/nomic-embed-text"]()
        ),
        persist_directory="./vector_db",
        verbose=False,
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
