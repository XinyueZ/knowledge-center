from typing import Callable, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel

from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.models.llms import llms_fn_lookup


def get_rag_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_fn_lookup["ChatCohere"]
        if USE_CLOUD_MODELS()
        else llms_fn_lookup["Ollama/command-r"]
    )


def get_smart_update_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_fn_lookup["Groq/mixtral-8x7b-32768"]
        if USE_CLOUD_MODELS()
        else llms_fn_lookup["Ollama/mixtral"]
    )


def get_put_readme_embed_llm_fn():
    return (
        (
            embeddings_fn_lookup["OpenAI/text-embedding-3-large"]
            if USE_CLOUD_MODELS()
            else embeddings_fn_lookup["Ollama/nomic-embed-text"]
        ),
        (
            llms_fn_lookup["Groq/mixtral-8x7b-32768"]
            if USE_CLOUD_MODELS()
            else llms_fn_lookup["Ollama/mistral"]
        ),
    )
