"""
data_loader.py: Document Loading & Ingestion Pipeline

Provides utilities to load, parse, and metadata-enrich policy documents (.pdf, .txt, .docx).
Supports single-file loading and directory-wide batch loading.

PDF parsing now goes through Docling instead of PyPDFLoader:
  - Docling's layout model resolves reading order before emitting text, so
    two-column pages come out in the right order instead of interleaved.
  - Docling's table-structure model (TableFormer) is used to detect tables
    and pull them out separately as HTML, instead of letting them get
    flattened into the surrounding paragraph text.

Each loaded PDF now yields two things:
  1. Normal per-page text Documents (tables excluded) for the existing
     chunking/vectorstore path -- unchanged downstream.
  2. A list of raw table dicts (HTML + caption + metadata, no summary yet)
     for the separate summarization/storage path in table_store.py.

Key Functions
-------------
- load_document():
    Loads a single document file, classifies its policy type, enriches metadata,
    and raises errors on failure. Returns {"documents": [...], "tables": [...]}.
    Used for dynamic single-file uploads (when adding documents via Streamlit UI).

- load_all_documents():
    Scans a directory recursively for all supported documents, delegating parsing
    to load_document(). Catches per-file errors to prevent a single corrupted file
    from halting batch processing. Used for initializing the knowledge base (startup
    indexing for CLI). Returns (documents, tables, failures).

Supported File Formats
----------------------
- PDF (.pdf) via Docling (text + tables)
- Plain Text (.txt) via TextLoader
- Word Documents (.docx) via Docx2txtLoader
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.policy_classifier import classify_document_policy_type

logger = logging.getLogger(__name__)


def _base_metadata(file_path: Path) -> Dict[str, Any]:
    stat = file_path.stat()
    return {
        "source": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_type": file_path.suffix.lower().lstrip("."),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _load_pdf(file_path: Path) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Parse a PDF with Docling.

    Returns (text_docs, raw_tables):
      - text_docs: one Document per page, tables excluded, with reading
        order already resolved across columns by Docling's layout model.
      - raw_tables: one dict per detected table -- {table_id, html, caption,
        metadata} -- with NO summary yet. table_store.build_table_documents()
        turns these into Documents (summary as page_content) later; this
        function only extracts, it doesn't call an LLM.

    NOTE: pipeline/table-mode tuning (e.g. TableFormerMode.ACCURATE vs FAST,
    forcing OCR) goes through DocumentConverter's format_options -- check
    your installed docling version's docs if you need to change it from the
    defaults used here.
    """
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.labels import DocItemLabel

    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    doc = result.document

    base_meta = _base_metadata(file_path)
    text_labels = set(DocItemLabel) - {DocItemLabel.TABLE}

    # --- per-page text, tables excluded, reading order already resolved ---
    text_docs: List[Document] = []
    num_pages = doc.num_pages()
    for page_no in range(1, num_pages + 1):
        page_text = doc.export_to_markdown(page_no=page_no, labels=text_labels).strip()
        if not page_text:
            continue
        text_docs.append(
            Document(page_content=page_text, metadata={**base_meta, "page": page_no})
        )

    if not text_docs:
        logger.warning("Docling extracted no text content from %s", file_path.name)

    # --- tables, extracted separately as HTML ---
    raw_tables: List[Dict[str, Any]] = []
    for idx, table in enumerate(doc.tables):
        try:
            caption = table.caption_text(doc)
        except Exception:
            caption = ""

        html = table.export_to_html(doc=doc, add_caption=True)
        page_no = table.prov[0].page_no if table.prov else None

        table_id = f"{file_path.stem}::table_{idx}"
        if page_no is not None:
            table_id += f"_p{page_no}"

        raw_tables.append(
            {
                "table_id": table_id,
                "html": html,
                "caption": caption,
                "metadata": {**base_meta, "page": page_no, "table_index": idx},
            }
        )

    return text_docs, raw_tables


def _load_txt(file_path: Path) -> Tuple[List[Document], List[Dict[str, Any]]]:
    try:
        docs = TextLoader(str(file_path), autodetect_encoding=True).load()
    except ImportError:
        docs = TextLoader(str(file_path), autodetect_encoding=False).load()
    for d in docs:
        d.metadata.update(_base_metadata(file_path))
    return docs, []


def _load_docx(file_path: Path) -> Tuple[List[Document], List[Dict[str, Any]]]:
    from langchain_community.document_loaders import Docx2txtLoader
    docs = Docx2txtLoader(str(file_path)).load()  # pyright: ignore[reportAbstractUsage] # type: ignore[abstract]
    for d in docs:
        d.metadata.update(_base_metadata(file_path))
    return docs, []


# NOTE: every loader now returns (documents, raw_tables) -- raw_tables is
# just [] for txt/docx. This is a signature change from the previous
# List[Any]-only loaders.
LOADER_REGISTRY = {".pdf": _load_pdf, ".txt": _load_txt, ".docx": _load_docx}


def load_document(file_path: Path, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[Any]]:
    """
    Load a single PDF/TXT/DOCX file: dispatch to the right loader, classify
    its policy_type from a text sample, and tag metadata accordingly.

    Returns {"documents": [Document, ...], "tables": [raw table dict, ...]}.
    "tables" is only ever non-empty for PDFs, and holds un-summarized HTML --
    pass it to table_store.build_table_documents() to get embeddable Documents.

    Raises on failure (unsupported extension, loader error, etc.) - callers
    decide how to handle/report it. load_all_documents() below catches and
    collects failures per-file; a single-file caller (e.g. an upload
    endpoint) can let it propagate and show the error directly.
    """
    suffix = file_path.suffix.lower()
    if suffix not in LOADER_REGISTRY:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(LOADER_REGISTRY)}"
        )

    loader_fn = LOADER_REGISTRY[suffix]
    loaded, raw_tables = loader_fn(file_path)

    sample_text = "\n".join(d.page_content for d in loaded[:3])
    policy_type = classify_document_policy_type(sample_text)
    if policy_type is None:
        logger.warning("Could not classify policy_type for %s; tagging as 'unknown'", file_path.name)

    for d in loaded:
        d.metadata["policy_type"] = policy_type or "unknown"
    for t in raw_tables:
        t["metadata"]["policy_type"] = policy_type or "unknown"

    if extra_metadata:
        for d in loaded:
            d.metadata.update(extra_metadata)
        for t in raw_tables:
            t["metadata"].update(extra_metadata)

    return {"documents": loaded, "tables": raw_tables}


def load_all_documents(
    data_dir: Union[str, Path], extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Load all PDF/TXT/DOCX files under data_dir (recursively).
    Returns (documents, tables, failures) -- failures is a list of {"file", "error"}
    """
    data_path = Path(data_dir).resolve()
    logger.info("Scanning %s for PDF/TXT/DOCX files", data_path)

    documents: List[Any] = []
    tables: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    files = [f for f in data_path.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_REGISTRY]
    logger.info("Found %d supported files", len(files))

    for file_path in files:
        try:
            result = load_document(file_path, extra_metadata=extra_metadata)
            documents.extend(result["documents"])
            tables.extend(result["tables"])
            logger.info(
                "Loaded %d doc(s), %d table(s) from %s",
                len(result["documents"]), len(result["tables"]), file_path.name,
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            failures.append({"file": str(file_path), "error": str(e)})

    logger.info(
        "Total: %d documents, %d tables loaded, %d failures",
        len(documents), len(tables), len(failures),
    )
    return documents, tables, failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs, tables, failed = load_all_documents("data")
    print(f"Loaded {len(docs)} documents, {len(tables)} tables.")
    if failed:
        print(f"{len(failed)} file(s) failed:")
        for f in failed:
            print(f"  {f['file']}: {f['error']}")
    if docs:
        print("Example metadata:", docs[0].metadata)