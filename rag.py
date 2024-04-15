from typing import Any, Dict

from langchain_cohere import ChatCohere
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

rag_llm = ChatCohere(model="command-r-plus", temperature=0)


def build_rag_chain(
    llm: BaseLanguageModel,
    preamble: str,
) -> str:
    struct_llm = llm.bind(preamble=preamble)
    prompt = lambda state: ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                state["prompt"],
                additional_kwargs={"documents": state["documents"]},
            )
        ]
    )
    return prompt | struct_llm | StrOutputParser()
