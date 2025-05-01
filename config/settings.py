import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "langchain_docs.csv")
PROCESSED_CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, "chunks.pkl")

# Model paths
VECTORSTORE_DIR = os.path.join(MODELS_DIR, "vectorstore")

# Other settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
EMBEDDING_MODEL = "all-MiniLM-L6-v2"