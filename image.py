import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import os
from pathlib import Path


def preprocess_image(image, size=(224, 224)):
    """Preprocess image for comparison"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard size
    image = cv2.resize(image, size)
    
    # Apply histogram equalization to normalize lighting
    image = cv2.equalizeHist(image)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

def calculate_similarity(img1, img2):
    """Calculate similarity between two images"""
    # Calculate SSIM between the two images
    similarity_score = ssim(img1, img2)
    return similarity_score * 100  # Convert to percentage

def load_and_preprocess_image(uploaded_file):
    """Load and preprocess uploaded image"""
    # Read image file
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    return processed_image, image

def main():
    st.set_page_config(page_title="Palm Image Similarity Analyzer", layout="wide")
    
    st.title("Palm Image Similarity Analyzer")
    st.markdown("""
    Upload a palm image and compare it with other palm images to find similarity percentages.
    The comparison uses advanced image processing techniques including structural similarity index.
    """)
    
    # Create directory for uploaded files if it doesn't exist
    upload_dir = Path("uploaded_palms")
    upload_dir.mkdir(exist_ok=True)
    
    # File uploader for main image
    main_image_file = st.file_uploader(
        "Upload your palm image", 
        type=['jpg', 'jpeg', 'png'],
        key="main_image"
    )
    
    # File uploader for comparison images
    st.markdown("### Upload Palm Images for Comparison (up to 5)")
    comparison_files = st.file_uploader(
        "Upload comparison images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="comparison_images"
    )
    
    if len(comparison_files) > 5:
        st.warning("Please upload a maximum of 5 comparison images.")
        comparison_files = comparison_files[:5]
    
    if main_image_file is not None and len(comparison_files) > 0:
        # Process main image
        main_processed, main_original = load_and_preprocess_image(main_image_file)
        
        st.markdown("### Main Palm Image")
        st.image(main_image_file, width=300)
        
        st.markdown("### Comparison Results")
        
        # Create columns for comparison images
        cols = st.columns(min(len(comparison_files), 5))
        
        # Process and compare each image
        for idx, (col, comp_file) in enumerate(zip(cols, comparison_files)):
            with col:
                # Process comparison image
                comp_processed, comp_original = load_and_preprocess_image(comp_file)
                
                # Calculate similarity
                similarity = calculate_similarity(main_processed, comp_processed)
                
                # Display comparison image and similarity
                st.image(comp_file, width=150)
                st.metric(
                    f"Similarity Score {idx + 1}",
                    f"{similarity:.1f}%",
                    help="Based on structural similarity index"
                )
                
                # Feature comparison
                if similarity > 70:
                    st.success("High similarity detected!")
                elif similarity > 40:
                    st.warning("Moderate similarity")
                else:
                    st.error("Low similarity")
        
        # Advanced Analysis Section
        if st.checkbox("Show Advanced Analysis"):
            st.markdown("### Advanced Analysis")
            
            # Create tabs for different analyses
            tab1, tab2 = st.tabs(["Edge Detection", "Contour Analysis"])
            
            with tab1:
                # Edge detection on main image
                edges = cv2.Canny(main_processed, 100, 200)
                st.image(edges, caption="Edge Detection Result", width=300)
                
            with tab2:
                # Find and draw contours
                contours, _ = cv2.findContours(
                    cv2.threshold(main_processed, 127, 255, cv2.THRESH_BINARY)[1],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contour_img = np.zeros_like(main_processed)
                cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
                st.image(contour_img, caption="Contour Analysis", width=300)

if __name__ == "__main__":
    main()