import logging
from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class RecursiveChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        documents = [d for d in documents if d.page_content and d.page_content.strip()]
        logger.info("Chunking %d documents", len(documents))
        chunks = self.splitter.split_documents(documents)
        chunks = [c for c in chunks if c.page_content.strip()]

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks
