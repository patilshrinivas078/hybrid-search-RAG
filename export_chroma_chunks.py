import json
import chromadb


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "policy_documents"
OUTPUT_FILE = "chroma_chunks.json"


def export_chroma_chunks():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    data = collection.get(include=["documents", "metadatas"])

    chunks = []

    for i, (document, metadata) in enumerate(zip(data["documents"], data["metadatas"])):
        chunks.append({
            "chunk_id": data["ids"][i],
            "text": document,
            "metadata": metadata or {}
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    export_chroma_chunks()