import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    def __init__(self):
        # Base directory setup (using parent.parent to go up two levels from config/__init__.py)
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.SRC_DIR = self.BASE_DIR / "src"
        
        # Data paths - now properly pointing to src/data
        self.DATA_DIR = self.SRC_DIR / "data"
        self.CACHE_DIR = self.DATA_DIR / "cache"  # Changed to be under DATA_DIR
        self.VECTOR_STORE_DIR = self.DATA_DIR / "vector_store"
        
        # Ensure directories exist (using makedirs which is more robust than mkdir)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)
        
        # Model settings (keeping your existing choices)
        self.EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Local embeddings
        self.LLM_MODEL = "gpt-3.5-turbo"
        
        # Vector store settings
        self.VECTOR_STORE_TYPE = "chroma"  # or "faiss"
        self.VECTOR_STORE_PATH = str(self.VECTOR_STORE_DIR)  # Using the dedicated dir
        
        # Document processing settings
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        
        # Retrieval settings
        self.TOP_K = 3
    
    @property
    def OPENAI_API_KEY(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return key

config = Config()
