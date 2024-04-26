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
from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.embeddings import embeddings_fn_lookup


def get_chunker_splitter_embedings_selection(
    chunk_overlap, chunk_size=None
) -> Dict[str, Tuple[Callable[[], TextSplitter], str, str]]:
    chunker_splitter_embedings_selection: Dict[
        str, Tuple[Callable[[], TextSplitter], str, str]
    ] = (
        {
            "SentenceWindowChunker": (
                lambda _=None: SentenceWindowChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["CohereEmbeddings"]()
                    ),
                ),
                "SentenceWindowNodeParser",
                "CohereEmbeddings",
            ),
            "RecursiveCharacterTextChunker": (
                lambda _=None: RecursiveCharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["CohereEmbeddings"](),
                ),
                "RecursiveCharacterTextSplitter",
                "CohereEmbeddings",
            ),
            "CharacterTextChunker": (
                lambda _=None: CharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["CohereEmbeddings"](),
                ),
                "CharacterTextSplitter",
                "CohereEmbeddings",
            ),
        }
        if USE_CLOUD_MODELS()
        else {
            "SentenceWindowChunker": (
                lambda _=None: SentenceWindowChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["Ollama/nomic-embed-text"]()
                    ),
                ),
                "SentenceWindowNodeParser",
                "Ollama/nomic-embed-text",
            ),
            "RecursiveCharacterTextChunker": (
                lambda _=None: RecursiveCharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["Ollama/nomic-embed-text"](),
                ),
                "RecursiveCharacterTextSplitter",
                "Ollama/nomic-embed-text",
            ),
            "CharacterTextChunker": (
                lambda _=None: CharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["Ollama/nomic-embed-text"](),
                ),
                "CharacterTextSplitter",
                "Ollama/nomic-embed-text",
            ),
        }
    )

    chunker_splitter_embedings_selection["SentenceTransformersTokenTextChunker"] = (
        lambda _=None: SentenceTransformersTokenTextChunker(
            chunk_overlap=chunk_overlap,
            embeddings=embeddings_fn_lookup["SentenceTransformerEmbeddings"](),
        ),
        "SentenceTransformersTokenTextSplitter",
        "SentenceTransformerEmbeddings",
    )
    return chunker_splitter_embedings_selection
