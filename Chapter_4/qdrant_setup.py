"""
qdrant_setup.py
-----------------
Qdrant setup using in-memory mode -- no Docker, no server, no network
download required. All Qdrant behavior (collections, points, payloads,
filtering, search) works identically in :memory: mode and Docker mode.

The ONLY difference between modes is one line in get_client():
  - Learning / no Docker : QdrantClient(":memory:")
  - Local Docker         : QdrantClient(url="http://localhost:6333")
  - Qdrant Cloud         : QdrantClient(url="https://...", api_key="...")

When you want persistence later:
    docker run -p 6333:6333 -v <your_path>/qdrant_storage:/qdrant/storage qdrant/qdrant
    Then swap get_client() to QdrantClient(url="http://localhost:6333").

Install: pip install qdrant-client
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "fd_knowledge_base"
VECTOR_SIZE = 384        # paraphrase-multilingual-MiniLM-L12-v2 output dim


def get_client() -> QdrantClient:
    """
    Returns a connected Qdrant client.

    Currently uses :memory: mode -- no Docker needed, zero setup,
    data is lost when the Python process restarts.

    To switch to persistent local Docker, replace with:
        return QdrantClient(url="http://localhost:6333")

    To switch to Qdrant Cloud, replace with:
        return QdrantClient(url="https://your-cluster.qdrant.io", api_key="YOUR_KEY")
    """
    return QdrantClient(":memory:")


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Creates the collection if it doesn't already exist.

    recreate=True drops and rebuilds it -- use this when changing embedding
    models or chunk schemas, never on a live collection with real data.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists -- skipping creation")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}' (vector size={VECTOR_SIZE})")


def verify_connection(client: QdrantClient) -> None:
    """Quick sanity check -- prints collection info if everything is working."""
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection   : {COLLECTION_NAME}")
    print(f"Vector size  : {info.config.params.vectors.size}")
    print(f"Points count : {info.points_count}")
    print(f"Status       : {info.status}")


if __name__ == "__main__":
    client = get_client()
    create_collection(client, recreate=False)
    verify_connection(client)