"""
table_store.py: persistence + summarization for tables extracted by Docling.

This is the second half of the "one more storage" piece of the
semi-structured design: data_loader.py's Docling path extracts raw tables
(HTML + caption + metadata, no summary). This module:

  1. Persists the raw HTML in a lightweight local key/value store
     (TableDocStore, SQLite-backed) keyed by table_id.
  2. Summarizes each table with an LLM.
  3. Returns one langchain Document per table, page_content = summary,
     metadata linking back to the table_id -- ready to flow through the
     existing chunking/vectorstore pipeline exactly like a text chunk.

At query time, vectorstore.ChromaVectorStore.resolve_full_content() uses the
table_id to pull the HTML back out of the TableDocStore and swap it in for
the summary before the generation LLM ever sees it. Retrieval matches on
the summary; generation reads the structure.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TableDocStore:
    """Simple persistent key/value store for raw table HTML, keyed by table_id."""

    def __init__(self, db_path: str = "table_store.sqlite"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tables (
                table_id TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                caption TEXT,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def put(self, table_id: str, html: str, caption: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tables (table_id, html, caption, metadata) VALUES (?, ?, ?, ?)",
            (table_id, html, caption, json.dumps(metadata or {})),
        )
        self._conn.commit()

    def get(self, table_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT html, caption, metadata FROM tables WHERE table_id = ?", (table_id,)
        ).fetchone()
        if row is None:
            return None
        html, caption, metadata_json = row
        return {"html": html, "caption": caption, "metadata": json.loads(metadata_json)}

    def close(self) -> None:
        self._conn.close()


def _build_summary_prompt(html: str, caption: str) -> str:
    # Plain concatenation rather than str.format()/f-string templating of a
    # pre-built template -- table HTML can legitimately contain literal
    # "{"/"}" characters (currency codes, JSON-looking cell text, etc.) and
    # we don't want that breaking prompt construction.
    return (
        "You are indexing a table extracted from a policy document for retrieval.\n"
        "Write a concise (3-5 sentence) summary of the table below so someone "
        "searching by topic, column names, or key values could find it. "
        "Explicitly mention the column headers and the kind of data/rows the "
        "table contains. Do not invent values that are not in the table.\n\n"
        f"Caption: {caption or '(none)'}\n\n"
        "Table (HTML):\n"
        f"{html}\n\n"
        "Summary:"
    )


def default_llm_fn(prompt: str) -> str:
    """
    Minimal OpenAI-backed summarizer, used only if no llm_fn is supplied.
    Swap this out for whatever LLM client the rest of the project already
    uses -- this exists so the module works standalone / for a quick test.
    """
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def summarize_table(html: str, caption: str, llm_fn: Callable[[str], str] = default_llm_fn) -> str:
    prompt = _build_summary_prompt(html, caption)
    try:
        return llm_fn(prompt)
    except Exception as e:
        logger.error("Table summarization failed: %s", e)
        # Don't let one bad table call fail the whole ingestion run -- the
        # HTML is already safely in table_store regardless of this.
        return f"Table{' - ' + caption if caption else ''} (summary generation failed; raw HTML retained)."


def build_table_documents(
    raw_tables: List[Dict[str, Any]],
    table_store: TableDocStore,
    llm_fn: Callable[[str], str] = default_llm_fn,
) -> List[Document]:
    """
    For each raw table dict (as produced by data_loader.py's Docling path):
      - persist the HTML in table_store, keyed by table_id
      - summarize it with llm_fn
      - return a Document(page_content=summary, metadata={..., content_type:
        "table", table_id: ...}), ready to be concatenated with regular text
        chunks and embedded.

    Typical usage, replacing the old single load_all_documents() -> chunk_documents()
    -> build_from_chunks() call:

        documents, raw_tables, failures = load_all_documents(data_dir)
        table_store = TableDocStore("table_store.sqlite")
        table_docs = build_table_documents(raw_tables, table_store)
        chunks = chunker.chunk_documents(documents + table_docs)
        vectorstore.build_from_chunks(chunks, embeddings)
    """
    table_docs: List[Document] = []
    for table in raw_tables:
        table_id = table["table_id"]
        html = table["html"]
        caption = table.get("caption", "")

        table_store.put(table_id, html=html, caption=caption, metadata=table["metadata"])

        summary = summarize_table(html, caption, llm_fn=llm_fn)
        doc_metadata = {**table["metadata"], "content_type": "table", "table_id": table_id}
        table_docs.append(Document(page_content=summary, metadata=doc_metadata))

    logger.info("Summarized and stored %d table(s)", len(table_docs))
    return table_docs