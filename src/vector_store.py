from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from config import config
import os
from typing import Optional

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = self._get_embedding_model()
        self.vector_store_path = config.VECTOR_STORE_PATH
        self.vector_store_type = config.VECTOR_STORE_TYPE.lower()

    def _get_embedding_model(self) -> Embeddings:
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    def create_vector_store(self, documents) -> VectorStore:
        """Create a new vector store from documents."""
        if self.vector_store_type == "faiss":
            return FAISS.from_documents(documents, self.embedding_model)
        elif self.vector_store_type == "chroma":
            return Chroma.from_documents(
                documents,
                self.embedding_model,
                persist_directory=self.vector_store_path,
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def load_vector_store(self) -> Optional[VectorStore]:
        """Load an existing vector store from disk."""
        if not os.path.exists(self.vector_store_path):
            return None

        if self.vector_store_type == "faiss":
            return FAISS.load_local(
                self.vector_store_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        elif self.vector_store_type == "chroma":
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_model,
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def save_vector_store(self, vector_store: VectorStore):
        """Save the vector store to disk."""
        if self.vector_store_type == "faiss":
            vector_store.save_local(self.vector_store_path)
        elif self.vector_store_type == "chroma":
            vector_store.persist()
