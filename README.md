Awsome Ingestion
=====

# Introduction

The purpose of this repository is that we demonstrate how to ingest data from various sources into Chroma vector storage. We combine different approaches using the Llama-Index and LangChain. The focus is not on determining the superiority of algorithms or libraries; rather, it serves as a demonstration of data ingestion into Chroma. Additionally, there are RAG classes to generate descriptions for different indices.

# Powered By

- 🦜️🔗 LangChain
- 🦙 LlamaIndex 
- <img src="https://user-images.githubusercontent.com/7164864/217935870-c0bc60a3-6fc0-4047-b011-7b4c59488c91.png" alt="Streamlit logo"  width="25"    style="margin-top:50px"></img> Streamlit
- <img src="https://user-images.githubusercontent.com/891664/227103090-6624bf7d-9524-4e05-9d2c-c28d5d451481.png" alt="Chroma logo"  width="25%"    style="margin-top:50px"></img>

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

# References

#### Hypothetical Document Embeddings(HyDE)  [paper](https://arxiv.org/pdf/2212.10496.pdf), [read](https://teetracker.medium.com/rag-with-hypothetical-document-embeddings-hyde-0edeca23f891) [code](https://github.com/XinyueZ/knowledge-center/blob/main/knowledge_center/rags/hyde.py)

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


