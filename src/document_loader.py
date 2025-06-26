import json
from typing import List, Dict, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    # Original methods (maintained for backward compatibility)
    def load_json_documents(self, file_path: str) -> List[Dict]:
        """Original method - loads JSON with basic error handling"""
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def process_documents(self, raw_docs: List[Dict]) -> List[Document]:
        """Original method - processes documents with basic structure"""
        documents = []
        for doc in raw_docs:
            page_content = doc.get("text", "") or doc.get("content", "") or str(doc)
            metadata = {k: v for k, v in doc.items() if k not in ["text", "content"]}
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Original method - splits documents with basic error handling"""
        return self.text_splitter.split_documents(documents)

    # New silent operation methods
    def _silent_load(self, file_path: Union[str, Path]) -> List[Dict]:
        """New internal method - fails completely silently"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception:
            return []

    def load_documents(self, file_path: Union[str, Path]) -> List[Document]:
        """New user-facing method - never shows errors"""
        raw_docs = self._silent_load(file_path)
        if not raw_docs:
            return []
            
        processed = []
        for doc in raw_docs:
            try:
                content = doc.get('content', '') or doc.get('text', '') or str(doc)
                metadata = {k: v for k, v in doc.items() 
                          if k.lower() not in ['content', 'text']}
                processed.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            except Exception:
                continue
        
        try:
            return self.text_splitter.split_documents(processed)
        except Exception:
            return processed

    # Combined utility method
    def load_and_process(self, file_path: Union[str, Path], silent: bool = False) -> List[Document]:
        """Unified method that works for both modes"""
        if silent:
            return self.load_documents(file_path)
        else:
            raw_docs = self.load_json_documents(file_path)
            processed = self.process_documents(raw_docs)
            return self.split_documents(processed)
