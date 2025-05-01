import os
from langchain_community.vectorstores import FAISS
from src.models.embeddings import get_embeddings
from src.data.loader import load_dataset
from src.data.processor import create_chunks
from config.settings import VECTORSTORE_DIR


def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get the vector store

    Args:
        chunks (list): The chunks to create the vector store from

    Returns:
        FAISS: The vector store
    """
    embeddings = get_embeddings()
    vector_store_path = VECTORSTORE_DIR

    if not os.path.exists(vector_store_path):
        print("Creating vector store")
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(vector_store_path)
    else:
        print("Loading vector store")
        vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    return vector_store


if __name__ == "__main__":

    print("Loading dataset...")
    df = load_dataset()
    print("Creating chunks...")
    chunks = create_chunks(df, save_chunks=True)
    print("Creating/Loading vector store...")
    vector_store = create_or_get_vector_store(chunks)
    print("Vector store created/loaded successfully!")
