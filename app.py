import asyncio
import os
import shutil
from datetime import datetime
from typing import List, Tuple

import nest_asyncio
import streamlit as st
from knowledge_center.chat import get_chat_llm_fn
from knowledge_center.chunkers import get_chunker_fn_selections
from knowledge_center.dashboard import get_smart_update_llm_fn
from knowledge_center.dashboard.description_crud import (
    connect_db, delete_description, genenerate_and_load_description,
    update_description_by_index)
from knowledge_center.file_loader import files_uploader
from knowledge_center.models.embeddings import embeddings_fn_lookup
from knowledge_center.rags.chat_rag import ChatRAG
from knowledge_center.rags.hyde import HyDE
from knowledge_center.rags.recursive_rag import RecursiveRAG
from knowledge_center.rags.sub_queries_rag import SubQueriesRAG
from knowledge_center.readme import get_put_readme_embed_llm_fn
from knowledge_center.search import get_search_llm_fn
from knowledge_center.utils import (CHUNK_OVERLAP_DEFAULT,
                                    CHUNK_OVERLAP_MIN_VALUE,
                                    CHUNK_SIZE_DEFAULT, CHUNK_SIZE_MIN_VALUE,
                                    INDEX_PERSIST_DIR, pretty_print)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain.base import LangChainLLM
from tqdm.asyncio import tqdm

nest_asyncio.apply()


st.set_page_config(layout="wide")


async def chunk_and_indexing(file_fullpath_list: List[str]) -> Tuple[str, str]:
    with st.sidebar:
        chunker_fn_selections = get_chunker_fn_selections()
        chunker_names = list(chunker_fn_selections.keys())
        chunker_selector = st.selectbox(
            "Chunker",
            chunker_names,
            index=0,
            key="splitter_selector",
        )
        chunk_size = None
        if chunker_fn_selections[chunker_selector][-1]:
            chunk_size = st.number_input(
                "chunk_size",
                value=CHUNK_SIZE_DEFAULT,
                min_value=CHUNK_SIZE_MIN_VALUE,
            )
        chunk_overlap = None
        if chunker_fn_selections[chunker_selector][-2]:
            chunk_overlap = st.number_input(
                "chunk_overlap",
                value=CHUNK_OVERLAP_DEFAULT,
                min_value=CHUNK_OVERLAP_MIN_VALUE,
            )

        chunker = chunker_fn_selections[chunker_selector][0](chunk_overlap, chunk_size)
        splitter_name = chunker_fn_selections[chunker_selector][1]
        embeddings_name = chunker_fn_selections[chunker_selector][2]
        first_filefullpath = file_fullpath_list[0]
        filename_noext = "_".join(
            os.path.splitext(os.path.basename(first_filefullpath))[0].split(" ")
        )
        index_name = st.text_input(
            "Index name", filename_noext, placeholder="index name"
        ).strip()
        if index_name is None or index_name == "":
            st.error("Please provide a name for the collection")
            return
        else:
            if os.path.exists(INDEX_PERSIST_DIR) and index_name in os.listdir(
                INDEX_PERSIST_DIR
            ):
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
                        persist_directory=INDEX_PERSIST_DIR,
                        index_name=index_name,
                    )
                st.success("Done!")
    return splitter_name, embeddings_name


async def dashboard(splitter_name: str, embeddings_name: str):
    if not os.path.exists(INDEX_PERSIST_DIR) or len(os.listdir(INDEX_PERSIST_DIR)) < 1:
        st.info("No index found")
        return
    index_fullpath_list = [
        os.path.join(INDEX_PERSIST_DIR, index_dir_name)
        for index_dir_name in os.listdir(INDEX_PERSIST_DIR)
        if index_dir_name
    ]

    with st.spinner("..."):
        description_list = await genenerate_and_load_description(
            os.path.join(INDEX_PERSIST_DIR),
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
                shutil.rmtree(os.path.join(INDEX_PERSIST_DIR, index_name))
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


async def readme_ui():
    def put_readme():
        embed_fn, llm_fn = get_put_readme_embed_llm_fn()
        query_res = RecursiveRAG(
            verbose=False,
            llm=LangChainLLM(llm_fn()),
            embeddings=LangchainEmbedding(embed_fn()),
            persist_directory="./knowledge_center/readme/vector_db",
            docs=SimpleDirectoryReader(input_files=["./README.md"]).load_data(),
        ).query(
            query="""Briefly introduce the repository, give use sections:
- list the required APIs, libraries in Bash Export Code Style.
- specify the setup requirements, pip and conda.
- approach to run
- list references.
- and so on...

Also list other stuffs, dependencies, 3rd parties supports mentioned in the content that you find important."""
        )
        st.write_stream(query_res.response_gen)

    st.subheader("About me")
    with st.spinner("## ..."):
        put_readme()


async def search_ui():
    if not os.path.exists(INDEX_PERSIST_DIR) or len(os.listdir(INDEX_PERSIST_DIR)) < 1:
        st.info("No index found")
        return
    index_list = [
        name
        for name in os.listdir(INDEX_PERSIST_DIR)
        if os.path.isdir(os.path.join(INDEX_PERSIST_DIR, name))
    ]
    index_name = st.selectbox(
        "Indices",
        index_list,
        index=0,
        key="search_rag_index_selector",
    )
    if query := st.text_area("Search...", key="search_query_input"):
        with st.spinner("Searching..."):
            search_rag = SubQueriesRAG(
                llm=LangChainLLM(get_search_llm_fn()()),
                verbose=True,
                streaming=True,
                persist_directory=INDEX_PERSIST_DIR,  # persist_directory/index_name1, persist_directory/index_name2, persist_directory/index_name3 ...
                index_name=index_name,
            )
            query_res = search_rag(message=query)
            st.write_stream(query_res.response_gen)


async def chat_ui():
    if not os.path.exists(INDEX_PERSIST_DIR) or len(os.listdir(INDEX_PERSIST_DIR)) < 1:
        st.info("No index found")
        return

    def _chat_index_selection_change():
        if "bot" in st.session_state:
            del st.session_state["bot"]
            pretty_print("bot", "flushed bot")
        if "messages" in st.session_state:
            del st.session_state["messages"]
            pretty_print("bot", "flushed messages")

    index_list = [
        name
        for name in os.listdir(INDEX_PERSIST_DIR)
        if os.path.isdir(os.path.join(INDEX_PERSIST_DIR, name))
    ]
    index_name = st.selectbox(
        "Indices",
        index_list,
        index=0,
        key="chat_index_selector",
        on_change=_chat_index_selection_change,
    )
    try:
        st.session_state["bot"] = (
            ChatRAG(
                llm=LangChainLLM(get_chat_llm_fn()()),
                verbose=True,
                streaming=True,
                persist_directory=INDEX_PERSIST_DIR,  # persist_directory/index_name1, persist_directory/index_name2, persist_directory/index_name3 ...
                index_name=index_name,
            )
            if "bot" not in st.session_state
            else st.session_state["bot"]
        )
    except Exception as e:
        st.error(f"No index found for: {e}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Write...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = st.session_state["bot"](prompt)
            try:
                content = st.write_stream(res.response_gen)
            except Exception as e:
                pretty_print("Cannot streaming", str(e))
                st.write(res.response)
                content = res.response
            pretty_print("content", content)
        st.session_state.messages.append({"role": "assistant", "content": content})


async def main():
    st.sidebar.header("Knowledge Center")
    file_fullpath_list = files_uploader("# Upload files")
    # pretty_print("File fullpath list", file_fullpath_list)

    splitter_embeddings = None
    with st.sidebar:
        if not (file_fullpath_list is None or len(file_fullpath_list) < 1):
            splitter_embeddings = await chunk_and_indexing(file_fullpath_list)
        else:
            st.info("Please upload files")
    tab_about, tab_search, tab_chat, tab_dashboard = st.tabs(
        ["About", "Search", "Chat", "Dashboard"]
    )
    with tab_about:
        await readme_ui()
    with tab_search:
        await search_ui()
    with tab_chat:
        await chat_ui()
    with tab_dashboard:
        await dashboard(*splitter_embeddings if splitter_embeddings else (None, None))


if __name__ == "__main__":
    asyncio.run(main())
