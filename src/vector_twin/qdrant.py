from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import numpy as np

load_dotenv()

def get_qdrant_client():
    """Get Qdrant client connection."""
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

def create_collection(client: QdrantClient):
    """Create Qdrant collection if it doesn't exist."""
    collection_name = "celebrities"
    vector_size = 512
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created successfully")
    except Exception as e:
        print(f"Collection '{collection_name}' already exists or error: {str(e)}")

def insert_image_embedding(client: QdrantClient, embedding, point_id, label, image_base64):
    """Insert image embedding into Qdrant."""
    client.upsert(
        collection_name="celebrities",
        points=[
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "label": label,
                    "image": image_base64
                }
            )
        ]
    )

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def get_top_k_similar_images(client: QdrantClient, query_vector, k: int = 1):
    """Search for similar images in Qdrant."""
    try:
        # Ensure query_vector is a list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
            
        search_result = client.search(
            collection_name="celebrities",
            query_vector=query_vector,
            limit=k
        )
        
        if not search_result:
            return []
            
        # Return just the first result's payload and score
        result = search_result[0]
        return [(result.payload["label"], result.score)]
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return [] 