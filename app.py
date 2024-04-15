import streamlit as st
from file_loader import files_uploader
from utils import pretty_print
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings

embedding = NVIDIAEmbeddings(model="nvolveqa_40k")
Settings.embed_model = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))


def langchain_fn():
    with st.sidebar:
        splitter_selector = st.selectbox(
            "Select splitter",
            [
                "RecursiveCharacterTextSplitter",
                "CharacterTextSplitter",
                "SentenceTransformersTokenTextSplitter",
            ],
            index=0,
        )

        def _RecursiveCharacterTextSplitter_fn() -> RecursiveCharacterTextSplitter:
            return RecursiveCharacterTextSplitter(
                chunk_size=st.number_input("chunk_size", value=1000, min_value=1000),
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
                length_function=len,
            )

        def _CharacterTextSplitter_fn() -> CharacterTextSplitter:
            return CharacterTextSplitter(
                chunk_size=st.number_input("chunk_size", value=1000, min_value=1000),
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
                length_function=len,
            )

        def _SentenceTransformersTokenTextSplitter_fn():
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=st.number_input(
                    "chunk_overlap", value=1000, min_value=1000
                ),
            )

        splitter_selector_map = {
            "RecursiveCharacterTextSplitter": _RecursiveCharacterTextSplitter_fn,
            "CharacterTextSplitter": _CharacterTextSplitter_fn,
            "SentenceTransformersTokenTextSplitter": _SentenceTransformersTokenTextSplitter_fn,
        }
        splitter_selector_map[splitter_selector]()


def llama_index_fn():
    pass


def main():
    st.title("Hello Streamlit!")
    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    if file_fullpath_list is None or len(file_fullpath_list) < 1:
        st.write("No file uploaded")
        return

    with st.sidebar:
        library_selector = st.selectbox(
            "Select library", ["LangChain", "Llama-Index"], index=0
        )
        library_selector_map = {
            "LangChain": langchain_fn,
            "Llama-Index": llama_index_fn,
        }
        library_selector_map[library_selector]()


if __name__ == "__main__":
    main()
