"""
debug_table_pipeline.py

A manual smoke test for the docling-based table/two-column pipeline.
Run this against a real PDF that has BOTH a two-column section and at
least one table, and read the printed output at each stage -- this
doesn't assert pass/fail, it shows the actual extracted content so
we can judge correctness ourselves.

Usage:
    python debug_table_pipeline.py /path/to/your_test_doc.pdf

"""

import sys
from pathlib import Path

from src.data_loader import load_document
from src.table_store import TableDocStore, build_table_documents
from src.chunking import RecursiveChunker


def main(pdf_path: str):
    pdf_path = Path(pdf_path)
    print(f"\n{'='*70}\nSTAGE 1: data_loader.load_document() -- extraction\n{'='*70}")
    result = load_document(pdf_path)
    docs, raw_tables = result["documents"], result["tables"]

    print(f"\n{len(docs)} page(s) of text, {len(raw_tables)} table(s) found.\n")

    print("--- Page text (first 300 chars each) ---")
    print("CHECK for any two-column page, the text should read in normal sentence order -- NOT column-1-line, column-2-line.")

    for d in docs:
        print(f"[page {d.metadata['page']}] policy_type={d.metadata['policy_type']}")
        print(d.page_content[:300].replace("\n", " ⏎ "))
        print("-" * 40)

    print("\n--- Raw tables ---")
    print("CHECK the HTML should have the right number of rows/columns and correct cell values.")
    for t in raw_tables:
        print(f"table_id={t['table_id']}  page={t['metadata']['page']}")
        print(f"caption: {t['caption']!r}")
        print(f"html: {t['html'][:500]}")
        print("-" * 40)

    if not raw_tables:
        print("\n!! No tables detected. Either this PDF has none, or docling'stable-structure model isn't firing.")

    print(f"\n{'='*70}\nSTAGE 2: table_store -- summarization + storage\n{'='*70}")
    store = TableDocStore(db_path="verify_table_store.sqlite")
    table_docs = build_table_documents(raw_tables, store)  # uses default_llm_fn (OpenAI) unless you pass llm_fn=

    print("CHECK each summary should mention the actual column headers and what the table is about.")
    for td in table_docs:
        print(f"table_id={td.metadata['table_id']}")
        print(f"summary: {td.page_content}")
        print("-" * 40)

    print("CHECK  whether table_store.get(table_id) returns the SAME html as stage 1.")
    for td in table_docs:
        record = store.get(td.metadata["table_id"])
        assert record is not None, f"MISSING from store: {td.metadata['table_id']}"
        original = next(t for t in raw_tables if t["table_id"] == td.metadata["table_id"])
        assert record["html"] == original["html"], "HTML round-trip mismatch!"
    print(f"round-trip OK for all {len(table_docs)} table(s)")

    print(f"\n{'='*70}\nSTAGE 3: chunking -- table docs pass through unsplit\n{'='*70}")
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(docs + table_docs)

    n_table_chunks = sum(1 for c in chunks if c.metadata.get("content_type") == "table")
    print(f"{len(chunks)} total chunks, {n_table_chunks} of them table chunks.")
    print("CHECK: n_table_chunks should exactly equal the number of tables")
    print(f"       found in stage 1 ({len(raw_tables)}). If it's higher, a table")
    print("       summary got split by the character; if")
    print("       lower, a table doc got dropped somewhere.\n")
    assert n_table_chunks == len(raw_tables), (
        f"Expected {len(raw_tables)} table chunks, got {n_table_chunks} -- "
        "a table summary was likely split or dropped."
    )
    print("table chunk count matches table count: OK")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_table_pipeline.py /path/to/test_doc.pdf")
        sys.exit(1)
    main(sys.argv[1])