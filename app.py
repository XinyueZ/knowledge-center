import streamlit as st
from file_loader import files_uploader
from utils import pretty_print


def main():
    st.title("Hello Streamlit!")
    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)

    if file_fullpath_list is None or len(file_fullpath_list) < 1:
        st.write("No file uploaded")
        return
    


if __name__ == "__main__":
    main()
