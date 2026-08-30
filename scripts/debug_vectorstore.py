"""
Debug script for ChromaVectorStore.

Run from the project root:
    python scripts/debug_vectorstore.py

Optionally pass a custom query:
    python scripts/debug_vectorstore.py "What is covered under IMT 22A?"
"""

import sys
import logging
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_all_documents
from src.chunking import RecursiveChunker
from src.embeddings import EmbeddingModel
from src.vectorstore import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# -- Config ------------------------------------------------------------------
DATA_DIR   = "data"
CHROMA_DIR = "chroma_db"
TOP_K      = 10
QUERY      = sys.argv[1] if len(sys.argv) > 1 else "What is covered under endorsement IMT 22A in two-wheeler policy?"

# -- Pipeline ----------------------------------------------------------------
print(f"\nQuery: {QUERY}\n" + "-" * 60)

docs, failures = load_all_documents(DATA_DIR)
if failures:
    print(f"Warning: Failed to load {len(failures)} document(s): {failures}")

chunker    = RecursiveChunker()
chunks     = chunker.chunk_documents(docs)

embedder   = EmbeddingModel()
embeddings = embedder.embed_chunks(chunks)

store = ChromaVectorStore(CHROMA_DIR)
store.build_from_chunks(chunks, embeddings)
store.load()

query_embedding = embedder.embed_query(QUERY)
results         = store.query(query_embedding, top_k=TOP_K)

# -- Output ------------------------------------------------------------------
for i, r in enumerate(results, 1):
    print(f"\n[{i}] distance={r['distance']:.4f}  id={r['id']}")
    print(f"    source: {r['metadata'].get('source', 'N/A')}")
    print(f"    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
