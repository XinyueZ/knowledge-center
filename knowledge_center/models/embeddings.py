from typing import Callable, Dict

from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings

from knowledge_center.models import USE_CLOUD_MODELS

embeddings_fn_lookup: Dict[str, Callable[[], Embeddings]] = (
    {
        "NVIDIAEmbeddings": lambda: NVIDIAEmbeddings(model="nvolveqa_40k"),
        "CohereEmbeddings": lambda: CohereEmbeddings(model="embed-english-light-v3.0"),
        "SentenceTransformerEmbeddings": lambda: SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ),
        "OpenAI/text-embedding-3-large": lambda: OpenAIEmbeddings(
            model_name="text-embedding-3-large"
        ),
        "OpenAI/text-embedding-3-small": lambda: OpenAIEmbeddings(
            model_name="text-embedding-3-small"
        ),
        "OpenAI/text-embedding-ada-002": lambda: OpenAIEmbeddings(
            model_name="text-embedding-ada-002"
        ),
    }
    if USE_CLOUD_MODELS()
    else {
        "SentenceTransformerEmbeddings": lambda: SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ),
        "Ollama/nomic-embed-text": lambda: OllamaEmbeddings(
            model="nomic-embed-text:latest"
        ),
        "Ollama/llama2": lambda: OllamaEmbeddings(model="llama2:latest"),
        "Ollama/llama3": lambda: OllamaEmbeddings(model="llama3:latest"),
    }
)
