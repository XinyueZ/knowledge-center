from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from typing import Callable, Dict, Tuple
from chunkers.character_text_chunker import CharacterTextChunker
from chunkers.recursive_character_text_chunker import RecursiveCharacterTextChunker
from chunkers.sentence_transformers_token_text_chunker import (
    SentenceTransformersTokenTextChunker,
)
from chunkers.sentence_window_chunker import SentenceWindowChunker

from langchain_text_splitters.base import TextSplitter

CHUNK_SIZE_DEFAULT = 1000
CHUNK_SIZE_MIN_VALUE = 1000
CHUNK_OVERLAP_DEFAULT = 0
CHUNK_OVERLAP_MIN_VALUE = 0

default_embeddings = NVIDIAEmbeddings(model="nvolveqa_40k")
sentence_transformer_embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
embeddings_selection: [str, Embeddings] = {
    "NVIDIAEmbeddings": default_embeddings,
    "SentenceTransformerEmbeddings": sentence_transformer_embeddings,
}


def get_chunker_and_embedings_selection(
    chunk_overlap, chunk_size=None
) -> Dict[str, Tuple[Callable[[], TextSplitter], str]]:
    chunker_and_embedings_selection: Dict[
        str, Tuple[Callable[[], TextSplitter], str]
    ] = {
        "RecursiveCharacterTextChunker": (
            lambda _=None: RecursiveCharacterTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_selection["NVIDIAEmbeddings"],
            ),
            "NVIDIAEmbeddings",
        ),
        "CharacterTextChunker": (
            lambda _=None: CharacterTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_selection["NVIDIAEmbeddings"],
            ),
            "NVIDIAEmbeddings",
        ),
        "SentenceTransformersTokenTextChunker": (
            lambda _=None: SentenceTransformersTokenTextChunker(
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_selection["SentenceTransformerEmbeddings"],
            ),
            "SentenceTransformerEmbeddings",
        ),
    }
    return chunker_and_embedings_selection
