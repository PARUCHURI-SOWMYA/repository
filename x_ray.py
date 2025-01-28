import streamlit as st
from PIL import Image
import numpy as np
import random
import io

# Create the Streamlit interface
st.title('Brain Tumor Detection using X-ray Images')
st.write("Upload an X-ray image for analysis.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Perform a mock prediction when the button is clicked
    if st.button("Detect Tumor"):
        st.write("Classifying the image...")

        # Simulate a random prediction (either "No Tumor" or "Tumor Detected")
        result = random.choice(["No Tumor", "Tumor Detected"])
        confidence = random.uniform(0.75, 1.0) if result == "Tumor Detected" else random.uniform(0.85, 1.0)

        # Show result based on simulated prediction
        st.write(f"Result: {result} (Confidence: {confidence*100:.2f}%)")
