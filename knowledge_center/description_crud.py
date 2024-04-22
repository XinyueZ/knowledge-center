import os
import sqlite3
from datetime import datetime
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from tqdm.asyncio import tqdm

from knowledge_center.completions.vanilla_docs_query_chain import \
    VanillaDocsQueryChain
from knowledge_center.models.embeddings import embeddings_lookup
from knowledge_center.models.llms import llms_lookup


def connect_db() -> sqlite3.Connection:
    if not os.path.exists("./other_db"):
        os.makedirs("./other_db")
    conn = sqlite3.connect("./other_db/index_descriptions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS descriptions (
            index_name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            splitter_name TEXT NOT NULL,
            embeddings_name TEXT NOT NULL, 
            created_datetime TEXT NOT NULL
        );
    """
    )
    conn.commit()
    return conn


def insert_description(
    conn: sqlite3.Connection,
    index_name: str,
    description: str,
    splitter_name: str,
    embeddings_name: str,
):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO descriptions (index_name, description, splitter_name, embeddings_name, created_datetime)
        VALUES (?, ?, ?, ?, ?);
    """,
        (index_name, description, splitter_name, embeddings_name, datetime.now()),
    )
    conn.commit()


def has_index_description(conn: sqlite3.Connection, index_name: str):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    return cursor.fetchone() is not None


def delete_description(conn: sqlite3.Connection, index_name: str):
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    conn.commit()


def fetch_description_by_index(conn: sqlite3.Connection, index_name: str):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    return cursor.fetchone()


def update_description_by_index(
    conn: sqlite3.Connection, index_name: str, update_description: str
):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE descriptions SET description = ? WHERE index_name = ?",
        (update_description, index_name),
    )
    conn.commit()


def fetch_descriptions(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM descriptions")
    return cursor.fetchall()


async def _generate_description(
    conn: sqlite3.Connection,
    persist_directory: str,
    splitter_name: str,
    embeddings_name: str,
    index_fullpath: str,
) -> Tuple[str, str, str, str, str]:
    index_dir_name = os.path.basename(index_fullpath)
    if not has_index_description(conn, index_dir_name):
        embed = embeddings_lookup[embeddings_name]()
        saved_index = Chroma(
            collection_name=index_dir_name,
            persist_directory=os.path.join(persist_directory, index_dir_name),
            embedding_function=embed,
        )
        retriever: BaseRetriever = saved_index.as_retriever()
        chain = VanillaDocsQueryChain(llms_lookup["ChatCohere"]())
        description = chain(
            preamble="You're an AI assistant to get the description of the documents briefly.",
            documents=retriever.invoke("Get the description of the documents"),
        )
        insert_description(
            conn,
            index_dir_name,
            description,
            splitter_name,
            embeddings_name,
        )

    index_name, description, splitter_name, embeddings_name, created_datetime = (
        fetch_description_by_index(conn, index_dir_name)
    )
    return index_name, description, splitter_name, embeddings_name, created_datetime


async def genenerate_and_load_description(
    persist_directory: str,
    splitter_name,
    embeddings_name: str,
    index_fullpath_list: List[str],
) -> List[Tuple[str, str]]:
    conn = connect_db()
    tasks = [
        _generate_description(
            conn, persist_directory, splitter_name, embeddings_name, index_fullpath
        )
        for index_fullpath in index_fullpath_list
    ]

    all_descriptions = await tqdm.gather(*tasks)
    conn.close()
    return all_descriptions
