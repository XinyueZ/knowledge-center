import os
import sqlite3
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

from chunkers import embeddings_selection
from rag.vanilla_rag import VanillaRAG
from datetime import datetime

rag_selection = {"vanilla": VanillaRAG()}


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
            embeddings_name TEXT NOT NULL, 
            created_datetime TEXT NOT NULL
        );
    """
    )
    conn.commit()
    return conn


def insert_description(
    conn: sqlite3.Connection, index_name: str, description: str, embeddings_name: str
):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO descriptions (index_name, description, embeddings_name, created_datetime)
        VALUES (?, ?, ?, ?);
    """,
        (index_name, description, embeddings_name, datetime.now()),
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


def fetch_descriptions(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM descriptions")
    return cursor.fetchall()


def genenerate_and_load_description(
    persist_directory: str, embeddings_name: str, index_fullpath_list: List[str]
) -> List[Tuple[str, str]]:
    conn = connect_db()
    all_descriptions = []
    for index_fullpath in index_fullpath_list:
        index_dir_name = os.path.basename(index_fullpath)
        if not has_index_description(conn, index_dir_name):
            embed = embeddings_selection[embeddings_name]
            saved_index = Chroma(
                persist_directory=os.path.join(persist_directory, index_dir_name),
                embedding_function=embed,
            )
            retriever: BaseRetriever = saved_index.as_retriever()
            description = rag_selection["vanilla"](
                prompt="Description of the documents",
                preamble="You're an AI assistant to get the description of the documents briefly.",
                documents=retriever.invoke("Get the description of the documents"),
            )
            insert_description(conn, index_dir_name, description, embeddings_name)

        index_name, description, embeddings_name, created_datetime = (
            fetch_description_by_index(conn, index_dir_name)
        )
        all_descriptions.append(
            (index_name, description, embeddings_name, created_datetime)
        )
    conn.close()
    return all_descriptions


def del_description(index_name: str):
    conn = connect_db()
    delete_description(conn, index_name)
    conn.close()
