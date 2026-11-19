import uuid

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from footnote.chunker import Chunk, chunk_document
from footnote.loader import FOLDERS, NOTES_DIR, load_note

load_dotenv()

MODEL = "nvidia/nemotron-3-embed-1b"
VECTOR_SIZE = 2048
COLLECTION = "notes"
DB_PATH = "data/qdrant"


def load_chunks() -> list[Chunk]:
    chunks = []
    for folder in FOLDERS:
        for path in sorted((NOTES_DIR / folder).glob("*.md")):
            chunks += chunk_document(load_note(path))
    return chunks


def to_point(chunk: Chunk, vector: list[float]) -> PointStruct:
    # Same source + text -> same id, so re-indexing replaces a chunk instead of duplicating it.
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.source + chunk.text))
    return PointStruct(id=point_id, vector=vector, payload=chunk.model_dump())


def index_all(client: QdrantClient, embedder: NVIDIAEmbeddings) -> None:
    chunks = load_chunks()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    vectors = embedder.embed_documents([c.text for c in chunks])
    client.upsert(COLLECTION, points=[to_point(c, v) for c, v in zip(chunks, vectors)])
    print(f"embedded {len(chunks)} chunks; collection now holds {client.count(COLLECTION).count}")


def search(client: QdrantClient, embedder: NVIDIAEmbeddings, question: str, k: int = 4) -> None:
    hits = client.query_points(COLLECTION, query=embedder.embed_query(question), limit=k).points
    print(f"\nQ: {question}")
    for hit in hits:
        print(f"  {hit.score:.3f}  {hit.payload.get('title')} > {hit.payload.get('heading')}")


def main() -> None:
    client = QdrantClient(path=DB_PATH)
    embedder = NVIDIAEmbeddings(model=MODEL)
    index_all(client, embedder)
    search(client, embedder, "Why do chunks overlap?")
    search(client, embedder, "How does the HPA decide when to add pods?")


if __name__ == "__main__":
    main()
