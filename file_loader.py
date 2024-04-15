from typing import List
import streamlit as st
import os
from utils import pretty_print


def files_uploader(prompt: str, storage_dir: str = "./tmp") -> List[str]:
    """Upload multiple files and return their full path or access via st.session_state["file_fullpath_list"]"""
    with st.sidebar:
        uploaded_docs = st.file_uploader(
            prompt,
            key="files_uploader",
            accept_multiple_files=True,
        )
        if not uploaded_docs:
            st.session_state["file_fullpath_list"] = None
            pretty_print("No file uploaded")
            return None
        if uploaded_docs:
            pretty_print("Uploaded files", uploaded_docs)
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)
            file_fullpath_list = []
            for uploaded_doc in uploaded_docs:
                temp_file_path = os.path.join(storage_dir, f"{uploaded_doc.name}")
                with open(temp_file_path, "wb") as file:
                    file.write(uploaded_doc.getvalue())
                    file_name = uploaded_doc.name
                    pretty_print("Uploaded file", file_name)
                    uploaded_doc.flush()
                    uploaded_doc.close()
                file_fullpath_list.append(temp_file_path)
            all_same_files = (
                all(
                    [
                        file_name == st.session_state["file_fullpath_list"][idx]
                        for idx, file_name in enumerate(file_fullpath_list)
                    ]
                )
                if st.session_state.get("file_fullpath_list")
                else False
            )
            if all_same_files:
                pretty_print("Same files uploaded")
                return st.session_state["file_fullpath_list"]

            pretty_print("New files uploaded")
            st.session_state["file_fullpath_list"] = file_fullpath_list
            return st.session_state["file_fullpath_list"]
        return None
