import asyncio
import datetime
import os
from typing import Callable, Dict, Iterable, List

import nest_asyncio
from sqlalchemy import desc
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_text_splitters.base import TextSplitter
from llama_index.core import Settings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from tqdm.asyncio import tqdm

from file_loader import files_uploader
from utils import pretty_print

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
            st.error("Please provide a name for the collection")
            return
        else:
            if f"{index_name}.pkl" in os.listdir(DB_PATH):
                st.error("Duplicate index name")
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
        st.info("No index found")
        return

    # list all pkl (index file) files in the directory
    index_fullpath_list = [
        os.path.join(DB_PATH, index_file_name)
        for index_file_name in os.listdir(DB_PATH)
        if index_file_name.endswith(".pkl")
    ]
    description_list = []
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Index")
    with col2:
        st.write("Created At")
    with col3:
        st.write("Op")

    for index_fullpath in index_fullpath_list:
        index_filename_no_ext = os.path.basename(index_fullpath).split(".")[0]

        with col1:
            st.write("")
            st.write(index_filename_no_ext)
        with col2:
            st.write("")
            # date of creation of the index
            file_create_time = os.path.getctime(index_fullpath)
            # to human readable format
            file_create_time = datetime.datetime.fromtimestamp(
                file_create_time
            ).strftime("%Y-%m-%d %H:%M:%S")
            st.write(file_create_time)
        with col3:
            if st.button("DEL", key=f"{index_filename_no_ext}_delete"):
                os.remove(index_fullpath)
                os.remove(f"{DB_PATH}/{index_filename_no_ext}.faiss")
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
            st.info("Please upload files")
    dashboard()


if __name__ == "__main__":
    asyncio.run(main())
