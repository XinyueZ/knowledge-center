import os
import sqlite3
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from rag import build_rag_chain, rag_llm


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect("index_descriptions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS descriptions (
            index_name TEXT PRIMARY KEY,
            description TEXT NOT NULL
        );
    """
    )
    conn.commit()
    return conn


def insert_description(conn, index_name, description):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO descriptions (index_name, description)
        VALUES (?, ?);
    """,
        (index_name, description),
    )
    conn.commit()


def has_index_description(conn, index_name):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    return cursor.fetchone() is not None


def delete_description(conn, index_name):
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    conn.commit()


def fetch_description_by_index(conn, index_name):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT description FROM descriptions WHERE index_name = ?;
    """,
        (index_name,),
    )
    return cursor.fetchone()


def fetch_descriptions(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM descriptions")
    return cursor.fetchall()


def genenerate_and_load_description(
    embed: Embeddings, index_fullpath_list: List[str]
) -> List[str]:
    conn = connect_db()
    all_descriptions = []
    for index_fullpath in index_fullpath_list:
        index_dir_name = os.path.basename(index_fullpath)
        if not has_index_description(conn, index_dir_name):
            saved_index = FAISS.load_local(
                index_fullpath,
                embed,
                allow_dangerous_deserialization=True,
            )
            retriever: BaseRetriever = saved_index.as_retriever()
            chain = build_rag_chain(
                rag_llm,
                preamble="You're an AI assistant to get the description of the documents briefly.",
            )
            description = chain.invoke(
                {
                    "prompt": "Description of the documents",
                    "documents": retriever.invoke(
                        "Get the description of the documents"
                    ),
                }
            )
            insert_description(conn, index_dir_name, description)

        description = fetch_description_by_index(conn, index_dir_name)
        all_descriptions.append(description)
    conn.close()
    return all_descriptions


def del_description(index_name: str):
    conn = connect_db()
    delete_description(conn, index_name)
    conn.close()
