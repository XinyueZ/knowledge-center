from typing import Callable


from langchain_core.language_models.base import BaseLanguageModel

from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.llms import llms_lookup


def get_rag_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_lookup["ChatCohere"] if USE_CLOUD_MODELS() else llms_lookup["Ollama/command-r"]
    )

def get_smart_update_llm_fn() -> Callable[[], BaseLanguageModel]:
    return (
        llms_lookup["Groq/mixtral-8x7b-32768"] if USE_CLOUD_MODELS() else llms_lookup["Ollama/mistral"]
    )
