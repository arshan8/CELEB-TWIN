import logging
from functools import lru_cache
from uuid import uuid4

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def initialize_models() -> tuple[torch.device, MTCNN, InceptionResnetV1]:
    """Initializes and returns the required ML models and device for face recognition.
    
    This function sets up the device (CPU/GPU), initializes the MTCNN model for face detection,
    and loads a pre-trained InceptionResnetV1 model for generating face embeddings.
    
    Returns:
        tuple:
            device (torch.device): The torch device (CPU/GPU) to use for computations
            mtcnn (MTCNN): Initialized MTCNN model for face detection
            resnet (InceptionResnetV1): Pre-trained InceptionResnetV1 model for embeddings
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return device, mtcnn, resnet


def process_single_image(img, device: torch.device, mtcnn: MTCNN, resnet: InceptionResnetV1) -> np.ndarray:
    """Process a single image and generates its embedding.
    
    Args:
        img: The celebrity image (PIL Image)
        device: Torch device to use
        mtcnn: MTCNN model for face detection
        resnet: ResNet model for embedding generation
    """
    try:
        # Ensure image is PIL Image
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
            
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Detect face
        face = mtcnn(img)
        if face is None:
            raise ValueError("No face detected in image. Please ensure the image contains a clear, front-facing face.")
        
        # Generate embedding
        face = face.unsqueeze(0).to(device)
        embedding = resnet(face).detach().cpu().numpy()
        
        # Ensure embedding is 1D array
        embedding = embedding.flatten()
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    except ValueError as e:
        logger.error(f"Face detection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")
