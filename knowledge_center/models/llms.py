from typing import Callable, Dict

from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.language_models.base import BaseLanguageModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from knowledge_center.models import USE_CLOUD_MODELS

llms_fn_lookup: Dict[str, Callable[[], BaseLanguageModel]] = (
    {
        "ChatCohere": lambda: ChatCohere(
            model="command-r-plus",
            temperature=0,
        ),
        "Groq/llama3-8b-8192": lambda: ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
        ),
        "Groq/llama3-70b-8192": lambda: ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
        ),
        "Groq/llama2-70b-4096": lambda: ChatGroq(
            model="llama2-70b-4096",
            temperature=0,
        ),
        "Groq/mixtral-8x7b-32768": lambda: ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
        ),
        "Groq/gemma-7b-it": lambda: ChatGroq(
            model="gemma-7b-it",
            temperature=0,
        ),
        "Anthropic/claude-3-opus-20240229": lambda: ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0,
        ),
        "Anthropic/claude-3-sonnet-20240229": lambda: ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0,
        ),
        "Anthropic/claude-3-haiku-20240307": lambda: ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
        ),
        "OpenAI/gpt-4-turbo": lambda: ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
        ),
        "OpenAI/gpt-3.5-turbo-0125": lambda: ChatAnthropic(
            model="gpt-3.5-turbo-0125",
            temperature=0,
        ),
    }
    if USE_CLOUD_MODELS()
    else {
        "Ollama/codegemma": lambda: ChatOllama(
            model="codegemma:latest",
            temperature=0,
        ),
        "Ollama/gemma:instruct": lambda: ChatOllama(
            model="gemma:instruct",
            temperature=0,
        ),
        "Ollama/gemma": lambda: ChatOllama(
            model="gemma:latest",
            temperature=0,
        ),
        "Ollama/command-r": lambda: ChatOllama(
            model="command-r:latest",
            temperature=0,
        ),
        "Ollama/llama2": lambda: ChatOllama(
            model="llama2:latest",
            temperature=0,
        ),
        "Ollama/llama3": lambda: ChatOllama(
            model="llama3:latest",
            temperature=0,
        ),
        "Ollama/mistral": lambda: ChatOllama(
            model="mistral:latest",
            temperature=0,
        ),
        "Ollama/mixtral": lambda: ChatOllama(
            model="mixtral:latest",
            temperature=0,
        ),
    }
)
