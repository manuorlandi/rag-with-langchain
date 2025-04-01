import os
import polars as pl
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def load_dataset(dataset_name: str = "dataset.csv") -> pl.DataFrame:
    """
    Load the dataset from the csv file

    Args:
        dataset_name (str): The name of the dataset to load

    Returns:
        pl.DataFrame: The loaded dataset containing data gathered by langchain
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pl.read_csv(file_path)
    return df


def create_chunks(
    df: pl.DataFrame,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> list[pl.DataFrame]:
    """
    Create chunks from the dataset

    Args:
        df (pl.DataFrame): The dataset to create chunks from
        chunk_size (int): The size of the chunks to create
        chunk_overlap (int): The overlap between the chunks

    Returns:
        list[pl.DataFrame]: The list of chunks
    """
    text_chunks = DataFrameLoader(df, page_content_column="body").load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # add metadata to chucks to simplofy retrieval
    for doc in text_splitter:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\DESCRIPTION:{description}"
        final_content += f"\BODY: {content}\nURL:{url}"
        doc.page_content = final_content

    return text_chunks


def create_or_get_vector_store(chunks: pl.DataFrame) -> FAISS:
    """
    Create or get the vector store

    Args:
        chunks (pl.DataFrame): The chunks to create the vector store from

    Returns:
        FAISS: The vector store
    """
    pass


def get_conversation_chain(
    vector_store: FAISS,
    human_message: str,
    system_message: str,
) -> None:
    """
    Create a conversation chain using the vector store and messages

    Args:
        vector_store (FAISS): The vector store containing the embedded documents
        human_message (str): The message template to use for human inputs
        system_message (str): The system message to set the AI assistant's behavior

    Returns:
        ConversationChain: The configured conversation chain for question answering
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
