import os
import pickle
import polars as pl
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import (
    PROCESSED_DATA_DIR,
    PROCESSED_CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def create_chunks(
    df: pl.DataFrame,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    save_chunks: bool = False,
) -> list:
    """
    Create chunks from the dataset

    Args:
        df (pl.DataFrame): The dataset to create chunks from
        chunk_size (int): The size of the chunks to create
        chunk_overlap (int): The overlap between the chunks
        save_chunks (bool): Whether to save the chunks to a file
    Returns:
        list: The list of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    text_chunks = DataFrameLoader(
        df.to_pandas(),
        page_content_column="body"
    ).load_and_split(
        text_splitter=text_splitter
    )

    # add metadata to chunks to simplify retrieval
    for doc in text_chunks:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\nDESCRIPTION: {description}"
        final_content += f"\nBODY: {content}\nURL: {url}"
        doc.page_content = final_content

    if save_chunks:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(PROCESSED_CHUNKS_FILE, "wb") as f:
            pickle.dump(text_chunks, f)

    return text_chunks


def load_processed_chunks():
    """
    Load previously processed chunks from disk

    Returns:
        list: The list of document chunks
    """
    with open(PROCESSED_CHUNKS_FILE, "rb") as f:
        return pickle.load(f)