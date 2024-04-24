import asyncio
import json
import os
import shutil
from datetime import datetime
from typing import List, Tuple

import nest_asyncio
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain.base import LangChainLLM
from tqdm.asyncio import tqdm

from knowledge_center.chunkers import get_chunker_splitter_embedings_selection
from knowledge_center.dashboard import (get_put_readme_embed_llm_fn,
                                        get_smart_update_llm_fn)
from knowledge_center.dashboard.description_crud import (
    connect_db, delete_description, genenerate_and_load_description,
    update_description_by_index)
from knowledge_center.file_loader import files_uploader
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.rags.hyde import HyDE
from knowledge_center.rags.recursive_rag import RecursiveRAG
from knowledge_center.utils import (CHUNK_OVERLAP_DEFAULT,
                                    CHUNK_OVERLAP_MIN_VALUE,
                                    CHUNK_SIZE_DEFAULT, CHUNK_SIZE_MIN_VALUE,
                                    pretty_print)

nest_asyncio.apply()


st.set_page_config(layout="wide")


DB_PATH = "./vector_db"


async def chunk_and_indexing(file_fullpath_list: List[str]) -> Tuple[str, str]:
    with st.sidebar:
        chunker_selector = st.selectbox(
            "Chunker",
            [
                "RecursiveCharacterTextChunker",
                "CharacterTextChunker",
                "SentenceTransformersTokenTextChunker",
                "SentenceWindowChunker",
            ],
            index=0,
            key="splitter_selector",
        )
        chunk_size = None
        if "Sentence" not in chunker_selector:
            chunk_size = st.number_input(
                "chunk_size",
                value=CHUNK_SIZE_DEFAULT,
                min_value=CHUNK_SIZE_MIN_VALUE,
            )

        chunk_overlap = st.number_input(
            "chunk_overlap",
            value=CHUNK_OVERLAP_DEFAULT,
            min_value=CHUNK_OVERLAP_MIN_VALUE,
        )

        chunker_and_embedings_selection = get_chunker_splitter_embedings_selection(
            chunk_overlap, chunk_size
        )
        chunker = chunker_and_embedings_selection[chunker_selector][0]()
        splitter_name = chunker_and_embedings_selection[chunker_selector][1]
        embeddings_name = chunker_and_embedings_selection[chunker_selector][2]
        index_name = st.text_input("Index name", placeholder="index name").strip()
        if index_name is None or index_name == "":
            st.error("Please provide a name for the collection")
            return
        else:
            if os.path.exists(DB_PATH) and index_name in os.listdir(DB_PATH):
                st.error("Duplicate index name")
                return
            if st.button("Ingest", key="ingest_button"):
                with st.spinner("Chunk and indexing..."):
                    tasks = [
                        PyPDFLoader(filepath).aload() for filepath in file_fullpath_list
                    ]
                    docs_list = await tqdm.gather(
                        *tasks
                    )  # list of documents of each file
                    docs = [
                        doc for docs in docs_list for doc in docs
                    ]  # flatten the all documents

                    chunker(
                        documents=docs,
                        persist_directory=DB_PATH,
                        index_name=index_name,
                    )
                st.success("Done!")
    return splitter_name, embeddings_name


async def dashboard(splitter_name: str, embeddings_name: str):
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) < 1:
        return
    index_fullpath_list = [
        os.path.join(DB_PATH, index_dir_name)
        for index_dir_name in os.listdir(DB_PATH)
        if index_dir_name
    ]

    with st.spinner("..."):
        description_list = await genenerate_and_load_description(
            os.path.join(DB_PATH),
            splitter_name,
            embeddings_name,
            index_fullpath_list,
        )

    cols = [0.7, 3.0, 1.5, 1.5, 0.7]
    gap = "large"
    col1, col2, col3, col4, col5 = st.columns(cols, gap=gap)

    with col1:
        st.write("")
        st.markdown("#### Index")
    with col2:
        st.write("")
        st.markdown("#### Description")
    with col3:
        st.write("")
        st.markdown("#### Splitter")
    with col4:
        st.write("")
        st.markdown("#### Embedding Model")
    with col5:
        st.write("")
        st.markdown("#### Created At")

    sorted_description_list = sorted(
        description_list,
        key=lambda x: datetime.strptime(x[-1], "%Y-%m-%d %H:%M:%S.%f"),
        reverse=True,
    )
    st.write("---")
    for (
        index_name,
        description,
        splitter_name,
        embeddings_name,
        created_datetime,
    ) in sorted_description_list:
        col1, col2, col3, col4, col5 = st.columns(cols, gap=gap)

        with col1:
            st.subheader("")
            st.write(index_name)
            st.subheader("")

            def apply_delete(index_name: str):
                shutil.rmtree(os.path.join(DB_PATH, index_name))
                delete_description(connect_db(), index_name)

            if st.button(
                "🗑️",
                key=f"{index_name}_delete",
                help="Delete index",
                type="primary",
                on_click=apply_delete,
                args=[index_name],
            ):
                st.rerun()

        with col2:
            st.subheader("")
            st.write(description)

            def apply_smart_update(
                index_name: str, description: str, embeddings_name: str
            ):
                hyde = HyDE(
                    llm=LangChainLLM(get_smart_update_llm_fn()()),
                    embeddings=LangchainEmbedding(
                        embeddings_fn_lookup[embeddings_name]()
                    ),
                )
                res = hyde(
                    index_name=index_name,
                    persist_directory="./vector_db",
                    query=description,
                )
                update_description_by_index(connect_db(), index_name, str(res))

            st.button(
                "✨",
                help="smart update",
                key=f"{index_name}_smart_update",
                on_click=apply_smart_update,
                args=[index_name, description, embeddings_name],
            )
        with col3:
            st.subheader("")
            st.write(splitter_name)
        with col4:
            st.subheader("")
            st.write(embeddings_name)
        with col5:
            st.subheader("")
            created_datetime = datetime.fromisoformat(created_datetime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.write(created_datetime)

        st.write("---")


def put_readme():
    embed_fn, llm_fn = get_put_readme_embed_llm_fn()
    query_res = RecursiveRAG(
        llm=LangChainLLM(llm_fn()),
        embeddings=LangchainEmbedding(embed_fn()),
        docs=SimpleDirectoryReader(input_files=["./README.md"]).load_data(),
    ).query(
        query="""Briefly introduce the repository, give use sections:
1. list the required APIs in Bash Export Code Style.
2. specify the setup requirements, pip and conda."""
    )
    st.write_stream(query_res.response_gen)


async def show_readme():
    st.subheader("About me")
    with st.spinner("## ..."):
        put_readme()


async def main():
    st.sidebar.header("Knowledge Center")
    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    splitter_embeddings = None
    with st.sidebar:
        if not (file_fullpath_list is None or len(file_fullpath_list) < 1):
            splitter_embeddings = await chunk_and_indexing(file_fullpath_list)
        else:
            st.info("Please upload files")
    tab_dashboard, tab_chat = st.tabs(["Dashboard", "Chat"])
    with tab_dashboard:
        await dashboard(*splitter_embeddings if splitter_embeddings else (None, None))
        await show_readme()
    with tab_chat:
        st.write("Chat")


if __name__ == "__main__":
    asyncio.run(main())
