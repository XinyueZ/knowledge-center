import asyncio
import json
from typing import Iterable, List, Optional, Union

import yaml
from langchain_core.documents import Document as lc_Document
from langchain_core.embeddings import Embeddings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document as lli_Document
from llama_index.core.schema import IndexNode, TextNode
from llama_index.legacy.core.embeddings.base import BaseEmbedding

from knowledge_center.chunkers.base_chunker import BaseChunker
from knowledge_center.utils import lli_from_chroma_store

PROPOSITIONS_PROMPT = PromptTemplate(
    """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America." ]

Input: {node_text}
Output:"""
)


class DenseXRetrievalChunker(BaseChunker):
    proposition_llm: LLM
    workers: int
    text_splitter: TextSplitter

    def __init__(
        self,
        embeddings: Union[Embeddings, BaseEmbedding],
        proposition_llm: LLM,
        workers: int = 8,
    ) -> None:
        super().__init__(embeddings)
        self.proposition_llm = proposition_llm
        self.workers = workers

        self.text_splitter = SentenceSplitter()

    @classmethod
    def _cnvt(cls, lc_doc: lc_Document) -> lli_Document:
        lli_doc = lli_Document.from_langchain_format(lc_doc)
        return lli_doc

    def create_chunks(
        self, documents: Union[Iterable[lc_Document], Iterable[lli_Document]]
    ) -> Union[List[lc_Document], List[BaseNode]]:
        doc_0 = documents[0]
        if isinstance(doc_0, lc_Document):
            docs = list(map(DenseXRetrievalChunker._cnvt, documents))
        else:
            docs = documents
        nodes = self.text_splitter.get_nodes_from_documents(docs)
        sub_nodes = self._gen_propositions(nodes)
        all_nodes = nodes + sub_nodes
        return all_nodes

    def chunk(
        self,
        documents: Union[Iterable[lc_Document], Iterable[lli_Document]],
        persist_directory: str,
        index_name: str,
    ) -> None:
        chunks = self.create_chunks(documents)
        vector_store = lli_from_chroma_store(persist_directory, index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(
            chunks, storage_context=storage_context, embed_model=self.embeddings
        )

    async def _aget_proposition(self, node: TextNode) -> List[TextNode]:
        """Get proposition."""
        inital_output = await self.proposition_llm.apredict(
            PROPOSITIONS_PROMPT, node_text=node.text
        )
        outputs = inital_output.split("\n")

        all_propositions = []

        for output in outputs:
            if not output.strip():
                continue
            if not output.strip().endswith("]"):
                if not output.strip().endswith('"') and not output.strip().endswith(
                    ","
                ):
                    output = output + '"'
                output = output + " ]"
            if not output.strip().startswith("["):
                if not output.strip().startswith('"'):
                    output = '"' + output
                output = "[ " + output

            try:
                propositions = json.loads(output)
            except Exception:
                # fallback to yaml
                try:
                    propositions = yaml.safe_load(output)
                except Exception:
                    # fallback to next output
                    continue

            if not isinstance(propositions, list):
                continue

            all_propositions.extend(propositions)

        assert isinstance(all_propositions, list)
        nodes = [TextNode(text=prop) for prop in all_propositions if prop]

        return [IndexNode.from_text_node(n, node.node_id) for n in nodes]

    def _gen_propositions(self, nodes: List[TextNode]) -> List[TextNode]:
        """Get propositions."""
        sub_nodes = asyncio.run(
            run_jobs(
                [self._aget_proposition(node) for node in nodes],
                show_progress=True,
                workers=self.workers,
            )
        )

        # Flatten list
        return [node for sub_node in sub_nodes for node in sub_node]
