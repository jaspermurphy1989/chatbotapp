from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from config import config

class CustomRetriever(BaseRetriever):
    """Custom retriever that wraps a vector store."""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__()
        self.vector_store = vector_store

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        return self.vector_store.similarity_search(query, k=config.TOP_K)
