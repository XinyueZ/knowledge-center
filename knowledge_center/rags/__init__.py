from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.groq import Groq

embeddings = NVIDIAEmbeddings(model="nvolveqa_40k")


default_hyde_update_query_llm = Groq(
    model="llama3-8b-8192",
    temperature=0.5,
    timeout=60,
)

default_hyde_gen_llm = Groq(
    model="llama3-8b-8192",
    temperature=0.8,
    timeout=60,
)
default_hyde_synthesizer_llm = Groq(
    model="llama3-8b-8192",
    temperature=0.0,
    timeout=60,
)
default_hyde_embeddings = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))
