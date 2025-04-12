import logging
from typing import List, Optional

from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams  # type: ignore

from vector_twin.settings import settings  # type: ignore

logger = logging.getLogger(__name__)


def create_collection(
    qdrant_client: QdrantClient,
    collection_name: str = settings.QDRANT_COLLECTION_NAME,
    vector_dimensions: int = settings.QDRANT_VECTOR_DIMENSIONS
):
    """Creates a new collection in Qdrant if it doesn't exist."""
    try:
        if not qdrant_client.collection_exists(collection_name):
            logger.info(f"Creating collection '{collection_name}' with vector size {vector_dimensions}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dimensions,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")


def insert_image_embedding(
    qdrant_client: QdrantClient,
    img_embedding: List[float],
    img_id: str,
    img_label: str,
    collection_name: str = settings.QDRANT_COLLECTION_NAME
):
    """Inserts or updates an image embedding in the specified Qdrant collection."""
    if img_embedding is None or not isinstance(img_embedding, list):
        logger.error("Invalid image embedding provided.")
        return

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=img_id,
                    vector=img_embedding,
                    payload={"label": img_label}
                )
            ]
        )
        logger.info(f"Inserted image ID '{img_id}' into collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Error inserting image embedding: {e}")


def get_top_k_similar_images(
    qdrant_client: QdrantClient,
    query_embedding: Optional[List[float]],
    collection_name: str = settings.QDRANT_COLLECTION_NAME,
    k: int = 5
):
    """Retrieves k most similar images to the query embedding."""
    if query_embedding is None:
        logger.error("❌ No embedding provided for similarity search.")
        raise ValueError("Query embedding cannot be None.")

    if not isinstance(query_embedding, list) or not all(isinstance(x, float) for x in query_embedding):
        logger.error("❌ Embedding must be a list of floats.")
        raise TypeError("Query embedding must be a list of floats.")

    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k
        )
        logger.info(f"Found {len(results)} similar items from collection '{collection_name}'")
        return results
    except Exception as e:
        logger.error(f"Error retrieving similar images: {e}")
        return []
