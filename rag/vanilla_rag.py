from rag.base_rag import BaseRAG
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class VanillaRAG(BaseRAG):
    _rag_llm = ChatCohere(model="command-r-plus", temperature=0)

    def _build_rag_chain(
        self,
        preamble: str,
    ) -> RunnableSerializable:
        struct_llm = self._rag_llm.bind(preamble=preamble)
        prompt = lambda state: ChatPromptTemplate.from_messages(
            [
                HumanMessage(
                    state["prompt"],
                    additional_kwargs={"documents": state["documents"]},
                )
            ]
        )
        return prompt | struct_llm | StrOutputParser()

    def __call__(self, *args, **kwds) -> str:
        return self._build_rag_chain(
            preamble=kwds["preamble"],
        ).invoke(
            {
                "prompt": kwds["prompt"],
                "documents": kwds["documents"],
            }
        )
