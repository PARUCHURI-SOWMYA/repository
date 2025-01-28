import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the pre-trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define the image size that the model expects
IMG_SIZE = 224

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Define the function to predict brain tumor
def predict_tumor(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    return class_idx, predictions[0][class_idx]

# Create Streamlit UI
st.title('Brain Tumor Detection using X-ray Images')
st.write("Upload an X-ray image for analysis.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Predict tumor type
    if st.button("Detect Tumor"):
        st.write("Classifying the image...")
        class_idx, confidence = predict_tumor(img)
        
        if class_idx == 0:
            st.write(f"Result: No Tumor (Confidence: {confidence*100:.2f}%)")
        else:
            st.write(f"Result: Tumor Detected (Confidence: {confidence*100:.2f}%)")
