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

        # Table summaries (metadata["content_type"] == "table", produced by
        # table_store.build_table_documents) are short, already 1:1 with a
        # single table, and carry a table_id that the vector store and the
        # TableDocStore both key off of. Running them through the character
        # splitter would cut that link for no benefit -- pass them through
        # untouched and only split ordinary text documents.
        table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]
        text_docs = [d for d in documents if d.metadata.get("content_type") != "table"]

        logger.info(
            "Chunking %d text document(s); %d table summary doc(s) passed through unsplit",
            len(text_docs), len(table_docs),
        )
        chunks = self.splitter.split_documents(text_docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        num_text_chunks = len(chunks)
        chunks.extend(table_docs)

        logger.info(
            f"Split {len(text_docs)} documents into {num_text_chunks} text chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap}); "
            f"+{len(table_docs)} table chunk(s) = {len(chunks)} total"
        )
        return chunks