from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain_core.runnables import RunnableSerializable

from knowledge_center.rags.base_rag import BaseRAG


class VanillaRAG(BaseRAG):
    llm: BaseLanguageModel

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def _build_rag_chain(
        self,
        preamble: str = None,
    ) -> RunnableSerializable:
        if preamble:
            struct_llm = self.llm.bind(preamble=preamble)
            prompt = lambda state: ChatPromptTemplate.from_messages(
                [
                    HumanMessage(
                        state["prompt"],
                        additional_kwargs={"documents": state["documents"]},
                    )
                ]
            )
            result = prompt | struct_llm | StrOutputParser()
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    HumanMessagePromptTemplate.from_template(
                        "{prompt}\nContext or documents:{documents}"
                    )
                ]
            )
            result = prompt | self.llm | StrOutputParser()
        return result

    def __call__(self, *args, **kwds) -> str:
        return self._build_rag_chain(preamble=kwds.get("preamble")).invoke(
            {
                "prompt": kwds["prompt"],
                "documents": kwds["documents"],
            }
        )
