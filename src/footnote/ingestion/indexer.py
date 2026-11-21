import hashlib
import uuid

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from footnote.ingestion.chunker import CHUNK_OVERLAP, MAX_CHARS, Chunk, chunk_document
from footnote.core.config import get_client
from footnote.ingestion.loader import FOLDERS, NOTES_DIR, load_note
from footnote.core.schema import Document

load_dotenv()

MODEL = "nvidia/nemotron-3-embed-1b"
VECTOR_SIZE = 2048
COLLECTION = "notes"


def load_notes() -> dict[str, Document]:
    docs = {}
    for folder in FOLDERS:
        for path in sorted((NOTES_DIR / folder).glob("*.md")):
            doc = load_note(path)
            docs[doc.source] = doc

    # print([doc.source for doc in docs.values()], "\n")
    return docs


def note_hash(doc: Document) -> str:
    # Settings are part of the hash: changing the chunker or model re-indexes every note.
    settings = f"{MODEL}|{MAX_CHARS}|{CHUNK_OVERLAP}"
    return hashlib.sha256((settings + doc.text).encode("utf-8")).hexdigest()


def indexed_hashes(client: QdrantClient) -> dict[str, str]:
    """source -> doc_hash for every note currently in the collection."""
    hashes: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION, limit=256, offset=offset, with_payload=["source", "doc_hash"]
        )
        for p in points:
            hashes[p.payload["source"]] = p.payload.get("doc_hash", "")
        if offset is None:
            return hashes


def plan_changes(current: dict[str, str], indexed: dict[str, str]) -> tuple[list[str], list[str]]:
    """Compare source -> hash maps.

    Returns (to_embed, to_remove):
      to_embed  - sources that are new or changed (their old chunks are deleted first)
      to_remove - sources that were indexed but no longer exist as notes
    """
    to_embed = [s for s, h in current.items() if indexed.get(s) != h]
    to_remove = [s for s in indexed if s not in current]
    return to_embed, to_remove


def delete_source(client: QdrantClient, source: str) -> None:
    # Filter(must=[FieldCondition(...)]): payload field "source" must exactly equal this note
    # (must = AND; should = OR). FilterSelector then deletes every point matching that filter.
    match_source = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
    client.delete(COLLECTION, points_selector=FilterSelector(filter=match_source))


def chunk_id(chunk: Chunk) -> str:
    # Same source + text -> same id, so re-indexing replaces a chunk instead of duplicating it.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.source + chunk.text))


def to_point(chunk: Chunk, vector: list[float], doc_hash: str) -> PointStruct:
    return PointStruct(id=chunk_id(chunk), vector=vector, payload=chunk.model_dump() | {"doc_hash": doc_hash})


def index_all(client: QdrantClient, embedder: NVIDIAEmbeddings) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    docs = load_notes()
    current = {source: note_hash(doc) for source, doc in docs.items()}
    to_embed, to_remove = plan_changes(current, indexed_hashes(client))

    for source in to_embed + to_remove:
        delete_source(client, source)

    chunks = [c for source in to_embed for c in chunk_document(docs[source])]
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        points = [to_point(c, v, current[c.source]) for c, v in zip(chunks, vectors)]
        client.upsert(COLLECTION, points=points)

    unchanged = len(current) - len(to_embed)
    print(
        f"{len(to_embed)} notes embedded ({len(chunks)} chunks), {unchanged} unchanged, "
        f"{len(to_remove)} removed; collection holds {client.count(COLLECTION).count}"
    )


def main() -> None:
    """Ingest only: embed new or changed notes. Asking questions lives in cli.py."""
    client = get_client()
    embedder = NVIDIAEmbeddings(model=MODEL)
    index_all(client, embedder)


if __name__ == "__main__":
    main()
