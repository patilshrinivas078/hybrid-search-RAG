"""
Debug script for RecursiveChunker.

Run from project root:
    python scripts/debug_chunking.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_all_documents
from src.chunking import RecursiveChunker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

def main():
    docs, failures = load_all_documents("data")
    if failures:
        print(f"Warning: Failed to load {len(failures)} document(s): {failures}")

    chunker = RecursiveChunker()
    chunks = chunker.chunk_documents(docs)

    print(f"\nProduced {len(chunks)} chunks across {len(docs)} documents.\n" + "-" * 60)
    if chunks:
        print("Example Chunk 0 Content:")
        print(chunks[0].page_content[:300] + "...")
        print("\nExample Chunk 0 Metadata:")
        print(chunks[0].metadata)

if __name__ == "__main__":
    main()
