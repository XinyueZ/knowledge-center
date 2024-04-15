import streamlit as st
from file_loader import files_uploader
from utils import pretty_print


def main():
    st.title("Hello Streamlit!")
    file_fullpath_list = files_uploader("# Upload files")
    pretty_print("File fullpath list", file_fullpath_list)


if __name__ == "__main__":
    main()
