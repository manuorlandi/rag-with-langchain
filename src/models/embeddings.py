from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from config.settings import EMBEDDING_MODEL


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Get the embeddings model

    Args:
        model_name (str): The name of the model to use

    Returns:
        Embeddings: The embeddings model
    """
    # For OpenAI embeddings:
    # return OpenAIEmbeddings()

    # For HuggingFace embeddings:
    return HuggingFaceEmbeddings(model_name=model_name)
