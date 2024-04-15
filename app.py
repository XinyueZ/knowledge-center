from altair import Text
import streamlit as st
from file_loader import files_uploader
from utils import pretty_print
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from typing import Dict, Callable, Iterable, List
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

lc_embedding = NVIDIAEmbeddings(model="nvolveqa_40k")
Settings.embed_model = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))


def langchain_fn(file_fullpath_list: List[str]):
    with st.sidebar:
        splitter_selector = st.selectbox(
            "Splitter",
            [
                "RecursiveCharacterTextSplitter",
                "CharacterTextSplitter",
                "SentenceTransformersTokenTextSplitter",
            ],
            index=0,
            key="splitter_selector",
        )

        def _RecursiveCharacterTextSplitter_fn() -> RecursiveCharacterTextSplitter:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=st.number_input("chunk_size", value=1000, min_value=1000),
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
            )

        def _CharacterTextSplitter_fn() -> CharacterTextSplitter:
            return CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=st.number_input("chunk_size", value=1000, min_value=1000),
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
            )

        def _SentenceTransformersTokenTextSplitter_fn() -> (
            SentenceTransformersTokenTextSplitter
        ):
            return SentenceTransformersTokenTextSplitter.from_tiktoken_encoder(
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
            )

        splitter_selector_map: Dict[str, Callable[[], TextSplitter]] = {
            "RecursiveCharacterTextSplitter": _RecursiveCharacterTextSplitter_fn,
            "CharacterTextSplitter": _CharacterTextSplitter_fn,
            "SentenceTransformersTokenTextSplitter": _SentenceTransformersTokenTextSplitter_fn,
        }
        splitter = splitter_selector_map[splitter_selector]()
        if st.button("Save"):
            with st.spinner("Chunk and indexing..."):
                for filepath in file_fullpath_list:
                    docs = PyPDFLoader(filepath).load()
                    chunks = splitter.split_documents(documents=docs)
                    collection = FAISS.from_documents(chunks, lc_embedding)
                    file_name = filepath.split("/")[-1].split(".")[0]
                    collection.save_local(f"./db/{file_name}")
            st.success("Done!")


def llama_index_fn():
    pass


def main():
    st.title("Knowledge Center")

    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    if file_fullpath_list is None or len(file_fullpath_list) < 1:
        st.write("No file uploaded")
        return

    with st.sidebar:
        llm_library_selector = st.selectbox(
            "Library",
            ["LangChain", "Llama-Index"],
            index=0,
            key="llm_library_selector",
        )
        llm_library_selector_map: Dict[str, Callable[[Iterable[Document]]]] = {
            "LangChain": langchain_fn,
            "Llama-Index": llama_index_fn,
        }

        llm_library_selector_map[llm_library_selector](file_fullpath_list)


if __name__ == "__main__":
    main()
