import asyncio
from calendar import c
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import nest_asyncio
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from tqdm.asyncio import tqdm

from knowledge_center.chunkers import (
    CHUNK_OVERLAP_DEFAULT, CHUNK_OVERLAP_MIN_VALUE, CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN_VALUE, get_chunker_splitter_embedings_selection)
from knowledge_center.description_crud import (connect_db, del_description,
                                               genenerate_and_load_description,
                                               insert_description)
from knowledge_center.file_loader import files_uploader
from knowledge_center.rags.adaptive_rag import AdaptiveRAG
from knowledge_center.utils import pretty_print

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
        index_name = st.text_input(
            "Index name(required, Press Enter to Save)", placeholder="index name"
        ).strip()
        if index_name is None or index_name == "":
            st.error("Please provide a name for the collection")
            return
        else:
            if os.path.exists(DB_PATH) and index_name in os.listdir(DB_PATH):
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

                chunker(
                    documents=docs,
                    persist_directory=DB_PATH,
                    index_name=index_name,
                )
            st.success("Done!")
    return splitter_name, embeddings_name


def dashboard(splitter_name: str, embeddings_name: str):
    if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) < 1:
        st.info("No index found")
        return

    index_fullpath_list = [
        os.path.join(DB_PATH, index_dir_name)
        for index_dir_name in os.listdir(DB_PATH)
        if index_dir_name
    ]

    description_list = genenerate_and_load_description(
        os.path.join(DB_PATH),
        splitter_name,
        embeddings_name,
        index_fullpath_list,
    )
    pretty_print("Dashboard / Index fullpath list", index_fullpath_list)
    pretty_print("Description list", description_list)

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
            c1, c12, c2 = st.columns(3)
            if c1.button("🚽", key=f"{index_name}_delete", help="Delete index"):
                shutil.rmtree(os.path.join(DB_PATH, index_name))
                del_description(index_name)
                st.experimental_rerun()
            c12.write("")
            c2.button("✎", key=f"{index_name}_edit", help="Edit index", type="primary")
        with col2:
            st.subheader("")
            st.write(description)
            
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


async def main():
    st.sidebar.header("Knowledge Center")

    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    splitter_embeddings = None
    with st.sidebar:
        if not (file_fullpath_list is None or len(file_fullpath_list) < 1):
            approach_selection = st.radio(
                "Adaptive RAG process or step-by-step",
                [
                    "Adaptive RAG Process",
                    "Step-by-step",
                ],
                index=0,
                key="step_by_step",
            )
            if approach_selection == "Adaptive RAG Process":
                index_name = st.text_input(
                    "Index name(required, Press Enter to Save)",
                    placeholder="index name",
                ).strip()
                if index_name is None or index_name == "":
                    st.error("Please provide a name for the collection")
                else:
                    if os.path.exists(DB_PATH) and index_name in os.listdir(DB_PATH):
                        st.error("Duplicate index name")
                    else:
                        with st.spinner("Chunk and indexing..."):
                            apt_rag = AdaptiveRAG(
                                index_dir=os.path.join(DB_PATH, index_name)
                            )
                            apt_rag_ds_list = await apt_rag.load_docs(
                                file_fullpath_list
                            )
                            kwargs = {
                                "ds_list": apt_rag_ds_list,
                                "query": "Documents description",
                            }
                            res = await apt_rag(**kwargs)
                            pretty_print("Description", res.response)
                            db_cnn = connect_db()
                            insert_description(db_cnn, index_name, res.response)
                            db_cnn.close()
                        st.success("Done!")
            else:
                splitter_embeddings = await chunk_and_indexing(file_fullpath_list)
        else:
            st.info("Please upload files")

    dashboard(*splitter_embeddings if splitter_embeddings else (None, None))


if __name__ == "__main__":
    asyncio.run(main())
