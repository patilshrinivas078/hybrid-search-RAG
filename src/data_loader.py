import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

def _base_metadata(file_path: Path) -> Dict[str, Any]:
    stat = file_path.stat()
    return {
        "source": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_type": file_path.suffix.lower().lstrip("."),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _load_pdf(file_path: Path) -> List[Any]:
    docs = PyPDFLoader(str(file_path)).load()
    for d in docs:
        d.metadata.update(_base_metadata(file_path))
    return docs


def _load_txt(file_path: Path) -> List[Any]:
    try:
        docs = TextLoader(str(file_path), autodetect_encoding=True).load()
    except ImportError:
        docs = TextLoader(str(file_path), autodetect_encoding=False).load()
    for d in docs:
        d.metadata.update(_base_metadata(file_path))
    return docs


def _load_docx(file_path: Path) -> List[Any]:
    from langchain_community.document_loaders import Docx2txtLoader
    docs = Docx2txtLoader(str(file_path)).load()
    for d in docs:
        d.metadata.update(_base_metadata(file_path))
    return docs


LOADER_REGISTRY = {".pdf": _load_pdf, ".txt": _load_txt, ".docx": _load_docx}


def load_all_documents(data_dir: str, extra_metadata: Optional[Dict[str, Any]] = None,) -> Tuple[List[Any], List[Dict[str, str]]]:
    """
    Load all PDF/TXT/DOCX files under data_dir (recursively).
    Returns (documents, failures) --failures is a list of {"file", "error"}
    """
    data_path = Path(data_dir).resolve()
    logger.info("Scanning %s for PDF/TXT/DOCX files", data_path)

    documents: List[Any] = []
    failures: List[Dict[str, str]] = []

    files = [f for f in data_path.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_REGISTRY]
    logger.info("Found %d supported files", len(files))

    for file_path in files:
        loader_fn = LOADER_REGISTRY[file_path.suffix.lower()]
        try:
            loaded = loader_fn(file_path)
            if extra_metadata:
                for d in loaded:
                    d.metadata.update(extra_metadata)
            documents.extend(loaded)
            logger.info("Loaded %d doc(s) from %s", len(loaded), file_path.name)
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            failures.append({"file": str(file_path), "error": str(e)})

    logger.info("Total: %d documents loaded, %d failures", len(documents), len(failures))
    return documents, failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs, failed = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    if failed:
        print(f"{len(failed)} file(s) failed:")
        for f in failed:
            print(f"  {f['file']}: {f['error']}")
    if docs:
        print("Example metadata:", docs[0].metadata)