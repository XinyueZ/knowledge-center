from knowledge_center.models import USE_CLOUD_MODELS
from knowledge_center.models.llms import llms_fn_lookup


def get_chat_llm_fn():
    return (
        llms_fn_lookup["Groq/mixtral-8x7b-32768"]
        if USE_CLOUD_MODELS()
        else llms_fn_lookup["Ollama/mistral"]
    )
