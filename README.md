Awsome Ingestion
=====

# Introduction

The purpose of this repository is that we demonstrate how to ingest data from various sources into Chroma vector storage. We combine different approaches using the Llama-Index and LangChain. The focus is not on determining the superiority of algorithms or libraries; rather, it serves as a demonstration of data ingestion into Chroma. Additionally, there are RAG classes to generate descriptions for different indices.

# Powered By

- 🦜️🔗 LangChain
- 🦙 LlamaIndex
- <img src="https://user-images.githubusercontent.com/7164864/217935870-c0bc60a3-6fc0-4047-b011-7b4c59488c91.png" alt="Streamlit logo"  width="25"></img> Streamlit
- <img src="https://user-images.githubusercontent.com/891664/227103090-6624bf7d-9524-4e05-9d2c-c28d5d451481.png" alt="Chroma logo"  width="25%"></img>
- <img alt="ollama"  src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" width="5%" height="5%"> Ollama

##### [Awesome LLMs applications and experiments](https://github.com/XinyueZ/chat-your-doc)

A repository demonstrating various approaches for RAG, Chunking, and other LLM-related algorithms.

# Chunking Principle

- Chunking Necessity: Vector databases need documents split
into chunks for retrieval and prompt generation.
- Query Result Variability: The same query will return different
content depending on how the document is chunked.
- Even Size Chunks: The easiest way is to split the document into
roughly even size chunks. This can result in similar content
getting split across chunks.
- Chunking by Atomic Elements: By identifying atomic elements,
you can chunk by combining elements rather than splitting raw
text.
  - Results in more coherent chunks
  - Example: combining content under the same section header into the same chunk.

# RAG

In the experiment, RAG is used in many places, such as search, chat, and smart update of description. There are various implementations for RAG, in the `rags` package. They may not all be used, but can be frequently used as replacements.

<table  style="border-collapse: collapse;">
<tr><td><img alt="RAG"  src="https://github.com/XinyueZ/knowledge-center/blob/main/assets/rag.png" width="46%" height="45%"/></td><td><img alt="RAG"  src="https://github.com/XinyueZ/knowledge-center/blob/main/assets/rag2.png" width="60%" height="45%"/></td></tr>  
</table>



# Setup

### Conda

```bash
conda env create -n machinelearning -f environment.yml
conda activate machinelearning
```

### Pip

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py --server.port 8011 --server.enableCORS false
```

# References

##### [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496.pdf) (Hypothetical Document Embeddings(HyDE))

- [RAG with Hypothetical Document Embeddings(HyDE)](https://teetracker.medium.com/rag-with-hypothetical-document-embeddings-hyde-0edeca23f891)
- [code](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/rags/hyde.py)

##### [Advanced Retrieval-Augmented Generation: From Theory to LlamaIndex Implementation](https://towardsdatascience.com/advanced-retrieval-augmented-generation-from-theory-to-llamaindex-implementation-4de1464a9930)

- [code](https://github.com/XinyueZ/knowledge-center/blob/705bf55a6a31f54fce65fb1ef82fdd1fd8991764/knowledge_center/rags/recursive_rag.py)

##### [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)

- [code](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/rags/adaptive_rag.py)
- [standalone code](https://github.com/XinyueZ/chat-your-doc/blob/master/advanced/llamaindex_adaptive_rag.py)

##### [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648)

- [code](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/chunkers/dense_x_retrieval_chunker.py)
- [standalone notebook](https://github.com/XinyueZ/chat-your-doc/blob/master/notebooks/DenseXRetrieval.ipynb)

##### Multi or Subqueries

- [code](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/rags/sub_queries_rag.py)
- [RAG with Multi-Query pattern](https://teetracker.medium.com/rag-with-multi-query-pattern-7272deb3401a)
- [LangChain / Llama-Index: RAG with Multi-Query Retrieval](https://teetracker.medium.com/langchain-llama-index-rag-with-multi-query-retrieval-4e7df1a62f83)

# Ollama setup

You must install Ollama to active the local models.

- Check the [ollama_option.json](https://github.com/XinyueZ/knowledge-center/blob/main/ollama_option.json) file to turn on the local models.
- Check [llms.py](https://github.com/XinyueZ/knowledge-center/blob/705bf55a6a31f54fce65fb1ef82fdd1fd8991764/knowledge_center/models/llms.py) to find the local models.

# Model list

## Embeddings

Check file  [embeddings.py](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/models/embeddings.py) to find the embeddings models.

## LLMs

Check file  [llms.py](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/models/llms.py) to find the LLMs models.

# Model config

Different feature can use different models and we differentiate them in different parts.

| Title | __init__.py |
|-------|-------------|
| ingestion (chunking and indexing) | [`__init__.py`](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/chunkers/__init__.py) |
| search | [`__init__.py`](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/search/__init__.py) |
| chat | [`__init__.py`](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/chat/__init__.py) |
| readme | [`__init__.py`](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/readme/__init__.py) |
| dashboard | [`__init__.py`](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/dashboard/__init__.py) |

# Key exports for LLMs and Embeddings

```python
export SERPAPI_API_KEY="e7945........."
export OPENAI_API_KEY="sk-........."
export GROQ_API_KEY="gsk_........."
export ANTHROPIC_API_KEY="sk-ant-........."
export LANGCHAIN_API_KEY="ls__........."
export NVIDIA_API_KEY="nvapi-........."
export HUGGING_FACE_TOKEN="hf_........."
export COHERE_API_KEY="zFiHtBT........."
export CO_API_KEY="zFiHtBT........."
```

# Star History

<br>
<div align="left">

<img src="https://api.star-history.com/svg?repos=XinyueZ/knowledge-center&type=Date" width="500px">

</div>
<br>
