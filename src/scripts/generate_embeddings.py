import os
import sys
import uuid
import base64
from io import BytesIO

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_path)

from datasets import load_dataset
from tqdm import tqdm
from vector_twin.models import initialize_models, process_single_image
from vector_twin.qdrant import get_qdrant_client, create_collection, insert_image_embedding

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()
    create_collection(qdrant_client)
    
    # Load dataset and limit to 120 samples
    print("Loading dataset...")
    dataset = load_dataset("lansinuote/simple_facenet", split="train")
    dataset = dataset.shuffle(seed=42).select(range(120))  # Limit to 120 samples
    
    # Initialize models
    print("Initializing models...")
    device, mtcnn, resnet = initialize_models()
    
    # Process images and store embeddings
    print("Generating embeddings...")
    for row in tqdm(dataset):
        try:
            img_embedding = process_single_image(row["image"], device, mtcnn, resnet)
            # Convert image to base64
            img_base64 = image_to_base64(row["image"])
            # Store both embedding and image
            insert_image_embedding(qdrant_client, img_embedding, str(uuid.uuid4()), row['label'], img_base64)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue

if __name__ == "__main__":
    main() 