from typing import Callable, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel

from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.llms import llms_fn_lookup


def get_rag_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_fn_lookup["ChatCohere"]
        if USE_CLOUD_MODELS()
        else llms_fn_lookup["Ollama/command-r"]
    )


def get_smart_update_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_fn_lookup["OpenAI/gpt-3.5"]
        if USE_CLOUD_MODELS()
        else llms_fn_lookup["Ollama/mixtral"]
    )
