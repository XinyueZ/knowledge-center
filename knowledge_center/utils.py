import os
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.pretty import pprint

VERBOSE = True
SIM_TOP_K = 5
RERANK_TOP_K = 5
WIN_SZ = 3
N_MULTI_STEPS = 5


CHUNK_SIZE_DEFAULT = 1000
CHUNK_SIZE_MIN_VALUE = 1000
CHUNK_OVERLAP_DEFAULT = 0
CHUNK_OVERLAP_MIN_VALUE = 0


def pretty_print(title: str = "Untitled", content: Any = None):
    if not VERBOSE:
        return
    print()
    print(f"-- {title} --")
    pprint(content)


def lli_from_chroma_store(
    persist_directory: str, index_name: str
) -> BasePydanticVectorStore:
    path = os.path.join(persist_directory, index_name)
    db: ClientAPI = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_or_create_collection(index_name)
    return ChromaVectorStore(chroma_collection)
