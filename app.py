import asyncio
import datetime
import os
import shutil
from typing import Callable, Dict, Iterable, List

import nest_asyncio
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

from description_crud import del_description, genenerate_and_load_description
from file_loader import files_uploader
from utils import pretty_print

nest_asyncio.apply()


st.set_page_config(layout="wide")


DB_PATH = "./vector_db"
CHUNK_SIZE_DEFAULT = 1000
CHUNK_SIZE_MIN_VALUE = 1000
CHUNK_OVERLAP_DEFAULT = 1000
CHUNK_OVERLAP_MIN_VALUE = 1000

lc_embedding = NVIDIAEmbeddings(model="nvolveqa_40k")
Settings.embed_model = LangchainEmbedding(NVIDIAEmbeddings(model="nvolveqa_40k"))


async def chunk_and_indexing(file_fullpath_list: List[str]):
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
                chunk_size=st.number_input(
                    "chunk_size",
                    value=CHUNK_SIZE_DEFAULT,
                    min_value=CHUNK_SIZE_MIN_VALUE,
                ),
                chunk_overlap=st.number_input(
                    "chunk_overlap",
                    value=CHUNK_OVERLAP_DEFAULT,
                    min_value=CHUNK_OVERLAP_MIN_VALUE,
                ),
            )

        def _CharacterTextSplitter_fn() -> CharacterTextSplitter:
            return CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=st.number_input(
                    "chunk_size",
                    value=CHUNK_SIZE_DEFAULT,
                    min_value=CHUNK_SIZE_MIN_VALUE,
                ),
                chunk_overlap=st.number_input(
                    "chunk_overlap",
                    value=CHUNK_OVERLAP_DEFAULT,
                    min_value=CHUNK_OVERLAP_MIN_VALUE,
                ),
            )

        def _SentenceTransformersTokenTextSplitter_fn() -> (
            SentenceTransformersTokenTextSplitter
        ):
            return SentenceTransformersTokenTextSplitter.from_tiktoken_encoder(
                chunk_overlap=st.number_input(
                    "chunk_overlap",
                    value=CHUNK_OVERLAP_DEFAULT,
                    min_value=CHUNK_OVERLAP_MIN_VALUE,
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
            if os.path.exists(DB_PATH) and f"{index_name}.pkl" in os.listdir(DB_PATH):
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
                db.save_local(os.path.join(DB_PATH, index_name))
            st.success("Done!")


def dashboard():
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) < 1:
        st.info("No index found")
        return

    index_fullpath_list = [
        os.path.join(DB_PATH, index_dir_name)
        for index_dir_name in os.listdir(DB_PATH)
        if index_dir_name
    ]

    description_list = genenerate_and_load_description(
        lc_embedding, index_fullpath_list
    )
    pretty_print("Description list", description_list)

    col1, col2, col3, col4 = st.columns([0.7, 7, 1.3, 0.88], gap="large")

    with col1:
        st.subheader("Index")
    with col2:
        st.subheader("Description")
    with col3:
        st.subheader("Created At")
    with col4:
        st.subheader("Operation")

    # sort index_fullpath_list by created time
    index_fullpath_list = sorted(
        index_fullpath_list,
        key=lambda x: os.path.getctime(x),
        reverse=True,
    )
    st.write("---")
    for index_fullpath in index_fullpath_list:
        col1, col2, col3, col4 = st.columns([0.7, 7, 1.3, 0.88], gap="large")
        index_dir_name = os.path.basename(index_fullpath)
        with col1:
            st.write(index_dir_name)
        with col2:
            st.write(description_list[index_fullpath_list.index(index_fullpath)][0])
        with col3:
            file_create_time = os.path.getctime(index_fullpath)
            file_create_time = datetime.datetime.fromtimestamp(
                file_create_time
            ).strftime("%Y-%m-%d %H:%M:%S")
            st.write(file_create_time)
        with col4:
            if st.button("🚽", key=f"{index_dir_name}_delete"):
                shutil.rmtree(index_fullpath)
                del_description(index_dir_name)
                st.experimental_rerun()
        st.write("---")


async def main():
    st.sidebar.header("Knowledge Center")

    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    with st.sidebar:
        if not (file_fullpath_list is None or len(file_fullpath_list) < 1):
            await chunk_and_indexing(file_fullpath_list)
        else:
            st.info("Please upload files")

    dashboard()


if __name__ == "__main__":
    asyncio.run(main())
