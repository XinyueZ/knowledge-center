import os
import sys
from typing import Any, List

from llama_index.core import VectorStoreIndex
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.chat_engine.types import (BaseChatEngine,
                                                ChatResponseMode,
                                                StreamingAgentChatResponse)
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain.base import LangChainLLM

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)


from knowledge_center.dashboard.description_crud import (connect_db,
                                                         fetch_descriptions)
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup
from knowledge_center.rags.base_rag import BaseRAG
from knowledge_center.utils import VERBOSE, lli_from_chroma_store, pretty_print


class ChatRAG(BaseRAG):
    runner: AgentRunner

    def __init__(
        self,
        llm: BaseLLM,
        embeddings: EmbedType,
        persist_directory: str,
        streaming: bool = True,
        verbose=VERBOSE,
    ) -> None:
        llm = llm
        embeddings = embeddings
        persist_directory = persist_directory
        streaming = streaming

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
        engines: List[BaseChatEngine] = [
            index.as_query_engine(llm=llm, streaming=True) for index in index_list
        ]
        desc_fmt = "Useful for simple queries on the content that covers the following dedicated topic:\n{topic}\n"
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

        self.runner = AgentRunner.from_llm(
            tools=query_engine_tools,
            llm=llm,
            verbose=verbose,
        )

    def __call__(self, *args: Any, **kwds: Any) -> ChatResponseMode:
        self.runner.chat(message=kwds["message"])


def main():
    chat_rag = ChatRAG(
        llm=LangChainLLM(llms_fn_lookup["Ollama/mistral"]()),
        embeddings=LangchainEmbedding(
            embeddings_fn_lookup["Ollama/nomic-embed-text"]()
        ),
        persist_directory="./vector_db",
        verbose=False,
    )

    pretty_print("I say", "Briefing all the content you have.")
    res: StreamingAgentChatResponse = chat_rag.runner.stream_chat(
        "Briefing all the content you have"
    )
    res.print_response_stream()

    pretty_print("I say", "Briefing the 2nd content you have.")
    res: StreamingAgentChatResponse = chat_rag.runner.stream_chat(
        "Briefing the 2nd content you have."
    )
    res.print_response_stream()

    pretty_print("I say", "hey, my name is Bob.")
    res: StreamingAgentChatResponse = chat_rag.runner.stream_chat(
        "hey, my name is Bob."
    )
    res.print_response_stream()

    pretty_print("I say", "Repeat my name if you could have remembered.")
    res: StreamingAgentChatResponse = chat_rag.runner.stream_chat("What is my name?")
    res.print_response_stream()


if __name__ == "__main__":
    main()
