from typing import Callable, Dict, Tuple

from langchain_text_splitters.base import TextSplitter
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

from knowledge_center.chunkers.character_text_chunker import \
    CharacterTextChunker
from knowledge_center.chunkers.recursive_character_text_chunker import \
    RecursiveCharacterTextChunker
from knowledge_center.chunkers.sentence_transformers_token_text_chunker import \
    SentenceTransformersTokenTextChunker
from knowledge_center.chunkers.sentence_window_chunker import \
    SentenceWindowChunker
from knowledge_center.models.embeddings import embeddings_lookup

CHUNK_SIZE_DEFAULT = 1000
CHUNK_SIZE_MIN_VALUE = 1000
CHUNK_OVERLAP_DEFAULT = 0
CHUNK_OVERLAP_MIN_VALUE = 0


def get_chunker_splitter_embedings_selection(
    chunk_overlap, chunk_size=None
) -> Dict[str, Tuple[Callable[[], TextSplitter], str, str]]:
    chunker_splitter_embedings_selection: Dict[
        str, Tuple[Callable[[], TextSplitter], str, str]
    ] = {
        "RecursiveCharacterTextChunker": (
            lambda _=None: RecursiveCharacterTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_lookup["NVIDIAEmbeddings"](),
            ),
            "RecursiveCharacterTextSplitter",
            "NVIDIAEmbeddings",
        ),
        "CharacterTextChunker": (
            lambda _=None: CharacterTextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_lookup["NVIDIAEmbeddings"](),
            ),
            "CharacterTextSplitter",
            "NVIDIAEmbeddings",
        ),
        "SentenceTransformersTokenTextChunker": (
            lambda _=None: SentenceTransformersTokenTextChunker(
                chunk_overlap=chunk_overlap,
                embeddings=embeddings_lookup["SentenceTransformerEmbeddings"](),
            ),
            "SentenceTransformersTokenTextSplitter",
            "SentenceTransformerEmbeddings",
        ),
        "SentenceWindowChunker": (
            lambda _=None: SentenceWindowChunker(
                embeddings=LangchainEmbedding(embeddings_lookup["NVIDIAEmbeddings"]()),
            ),
            "SentenceWindowNodeParser",
            "NVIDIAEmbeddings",
        ),
    }
    return chunker_splitter_embedings_selection
