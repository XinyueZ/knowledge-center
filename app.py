import os
from regex import F
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
from tqdm.asyncio import tqdm
import nest_asyncio
import datetime
import asyncio

nest_asyncio.apply()

# wide screen config for st app
st.set_page_config(layout="wide")


DB_PATH = "./db"
lc_embedding = NVIDIAEmbeddings(model="nvolveqa_40k")
Settings.embed_model = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))


async def langchain_fn(file_fullpath_list: List[str]):
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
        index_name = st.text_input(
            "Index name(required, Press Enter to Save)", placeholder="index name"
        ).strip()
        if index_name is None or index_name == "":
            st.warning("Please provide a name for the collection")
            return
        else:
            if f"{index_name}.pkl" in os.listdir(DB_PATH):
                st.warning("Duplicate index name")
                return

            with st.spinner("Chunk and indexing..."):
                tasks = [
                    PyPDFLoader(filepath).aload() for filepath in file_fullpath_list
                ]
                docs_list = await tqdm.gather(*tasks)  # list of documents of each file
                docs = [
                    doc for docs in docs_list for doc in docs
                ]  # flatten the all documents
                chunks = splitter.split_documents(documents=docs)
                db = FAISS.from_documents(chunks, lc_embedding)
                db.save_local(f"{DB_PATH}", index_name=index_name)
            st.success("Done!")


async def llama_index_fn():
    pass


def dashboard():
    # check DB path exists or empty
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) < 1:
        st.warning("No index found")
        return

    # list all pkl (index file) files in the directory
    index_files = [file for file in os.listdir(DB_PATH) if file.endswith(".pkl")]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Index")
    with col2:
        st.write("Created At")
    with col3:
        st.write("Op")

    for filename in index_files:
        filename_no_ext = filename.split(".")[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("")
            st.write(filename_no_ext)
        with col2:
            st.write("")
            # date of creation of the index
            file_fullpath = f"{DB_PATH}/{filename}"
            file_create_time = os.path.getctime(file_fullpath)
            # to human readable format
            file_create_time = datetime.datetime.fromtimestamp(
                file_create_time
            ).strftime("%Y-%m-%d %H:%M:%S")
            st.write(file_create_time)
        with col3:
            if st.button(f"❌", key=f"{filename_no_ext}_delete"):
                os.remove(f"{DB_PATH}/{filename_no_ext}.pkl")
                os.remove(f"{DB_PATH}/{filename_no_ext}.faiss")
                st.experimental_rerun()


async def main():
    st.title("Knowledge Center")

    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    with st.sidebar:
        if not (file_fullpath_list is None or len(file_fullpath_list) < 1):
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

            await llm_library_selector_map[llm_library_selector](file_fullpath_list)
        else:
            st.warning("Please upload files")
    dashboard()


if __name__ == "__main__":
    asyncio.run(main())
