import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "gpt-3.5-turbo"  # Default model
    
    # Vector store settings
    VECTOR_STORE_TYPE = "faiss"  # or "chroma"
    VECTOR_STORE_PATH = str(CACHE_DIR / "vector_store")
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K = 3
    
    @property
    def OPENAI_API_KEY(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return key

config = Config()
