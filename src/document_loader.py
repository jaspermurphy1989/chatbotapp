import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def load_json_documents(self, file_path: str) -> List[Dict]:
        """Load documents from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def process_documents(self, raw_docs: List[Dict]) -> List[Document]:
        """Process raw documents into LangChain Documents."""
        documents = []
        for doc in raw_docs:
            # Customize this based on your JSON structure
            page_content = doc.get("text", "") or doc.get("content", "") or str(doc)
            metadata = {k: v for k, v in doc.items() if k not in ["text", "content"]}
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
