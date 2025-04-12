import os
import sys

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import streamlit as st
from vector_twin.qdrant import get_qdrant_client, get_top_k_similar_images
from vector_twin.models import initialize_models, process_single_image
import tempfile
import cv2
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="CelebTwin",
    page_icon="🎭",
    layout="centered"
)

# Initialize session state
if 'matched_celebrity' not in st.session_state:
    st.session_state.matched_celebrity = None
    
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = get_qdrant_client()
    
if 'models' not in st.session_state:
    st.session_state.models = initialize_models()

# Initialize models
device, mtcnn, resnet = initialize_models()

# Initialize Qdrant client
qdrant_client = get_qdrant_client()

# Main app
st.title("Celebrity Twin Finder")
st.write("Find out which celebrity you look like!")

# Add helpful tips
st.info("""
Tips for best results:
- Use a clear, well-lit photo
- Face should be clearly visible and front-facing
- Avoid sunglasses or face coverings
- Make sure your face is centered in the image
""")

# Image upload or webcam
option = st.radio("Choose input method:", ("Upload Image", "Take Photo"))

def process_and_match(image):
    """Process image and find matches."""
    try:
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image format. Please upload a valid image file.")
            
        # Process the image
        embedding = process_single_image(image, device, mtcnn, resnet)
        
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Failed to generate image embedding")
            
        # Search for similar images
        results = get_top_k_similar_images(qdrant_client, embedding, k=1)
        
        # Display results
        if results and len(results) > 0:
            label, score = results[0]
            st.subheader("Your Celebrity Lookalike:")
            st.write(f"{label} (Similarity: {score:.2f})")
            st.balloons()
        else:
            st.warning("No celebrity matches found!")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error("Please try again with a different image.")

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            process_and_match(image)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

else:  # Take Photo
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        try:
            # Read the image
            bytes_data = img_file_buffer.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
            st.image(image, caption="Captured Image", use_column_width=True)
            process_and_match(image)
        except Exception as e:
            st.error(f"Error capturing image: {str(e)}")