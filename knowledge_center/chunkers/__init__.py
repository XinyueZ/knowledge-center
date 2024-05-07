from typing import Callable, Dict, Tuple

from langchain_text_splitters.base import TextSplitter
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain.base import LangChainLLM

from knowledge_center.chunkers.character_text_chunker import \
    CharacterTextChunker
from knowledge_center.chunkers.dense_x_retrieval_chunker import \
    DenseXRetrievalChunker
from knowledge_center.chunkers.recursive_character_text_chunker import \
    RecursiveCharacterTextChunker
from knowledge_center.chunkers.sentence_transformers_token_text_chunker import \
    SentenceTransformersTokenTextChunker
from knowledge_center.chunkers.sentence_window_chunker import \
    SentenceWindowChunker
from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup


def get_chunker_fn_selections() -> (
    Dict[str, Tuple[Callable[[], TextSplitter], str, str, bool, bool]]
):
    chunker_splitter_embedings_selection: Dict[
        str, Tuple[Callable[[], TextSplitter], str, str]
    ] = (
        {
            "DenseXRetrievalChunker": (
                lambda chunk_overlap, chunk_size=None: DenseXRetrievalChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["OpenAI/text-embedding-3-large"]()
                    ),
                    proposition_llm=(LangChainLLM(llms_fn_lookup["OpenAI/gpt-3.5"]())),
                ),
                "DenseXRetrievalChunker",
                "OpenAI/text-embedding-3-large",
                False,
                False,  # chunk_overlap, chunk_size needed or not
            ),
            "SentenceWindowChunker": (
                lambda chunk_overlap, chunk_size=None: SentenceWindowChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["OpenAI/text-embedding-3-large"]()
                    ),
                ),
                "SentenceWindowNodeParser",
                "OpenAI/text-embedding-3-large",
                False,
                False,  # chunk_overlap, chunk_size needed or not
            ),
            "RecursiveCharacterTextChunker": (
                lambda chunk_overlap, chunk_size=None: RecursiveCharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["OpenAI/text-embedding-3-large"](),
                ),
                "RecursiveCharacterTextSplitter",
                "OpenAI/text-embedding-3-large",
                True,
                True,  # chunk_overlap, chunk_size needed or not
            ),
            "CharacterTextChunker": (
                lambda chunk_overlap, chunk_size=None: CharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["OpenAI/text-embedding-3-large"](),
                ),
                "CharacterTextSplitter",
                "OpenAI/text-embedding-3-large",
                True,
                True,  # chunk_overlap, chunk_size needed or not
            ),
        }
        if USE_CLOUD_MODELS()
        else {
            "DenseXRetrievalChunker": (
                lambda chunk_overlap, chunk_size=None: DenseXRetrievalChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["OpenAI/text-embedding-3-large"]()
                    ),
                    proposition_llm=LangChainLLM(llms_fn_lookup["Ollama/mistral"]()),
                ),
                "DenseXRetrievalChunker",
                "OpenAI/text-embedding-3-large",
                False,
                False,  # chunk_overlap, chunk_size needed or not
            ),
            "SentenceWindowChunker": (
                lambda chunk_overlap, chunk_size=None: SentenceWindowChunker(
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup["Ollama/nomic-embed-text"]()
                    ),
                ),
                "SentenceWindowNodeParser",
                "Ollama/nomic-embed-text",
                False,
                False,  # chunk_overlap, chunk_size needed or not
            ),
            "RecursiveCharacterTextChunker": (
                lambda chunk_overlap, chunk_size=None: RecursiveCharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["Ollama/nomic-embed-text"](),
                ),
                "RecursiveCharacterTextSplitter",
                "Ollama/nomic-embed-text",
                True,
                True,  # chunk_overlap, chunk_size needed or not
            ),
            "CharacterTextChunker": (
                lambda chunk_overlap, chunk_size=None: CharacterTextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings_fn_lookup["Ollama/nomic-embed-text"](),
                ),
                "CharacterTextSplitter",
                "Ollama/nomic-embed-text",
                True,
                True,  # chunk_overlap, chunk_size needed or not
            ),
        }
    )

    chunker_splitter_embedings_selection["SentenceTransformersTokenTextChunker"] = (
        lambda chunk_overlap, chunk_size=None: SentenceTransformersTokenTextChunker(
            chunk_overlap=chunk_overlap,
            embeddings=embeddings_fn_lookup["SentenceTransformerEmbeddings"](),
        ),
        "SentenceTransformersTokenTextSplitter",
        "SentenceTransformerEmbeddings",
        True,
        False,  # chunk_overlap, chunk_size needed or not
    )
    return chunker_splitter_embedings_selection
