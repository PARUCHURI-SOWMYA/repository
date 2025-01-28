import streamlit as st
from PIL import Image
import numpy as np
import io

# Attempt to import TensorFlow and handle any import errors
try:
    import tensorflow as tf
    st.write("TensorFlow successfully imported!")
except ModuleNotFoundError:
    st.write("TensorFlow is not installed. Please install it by running `pip install tensorflow`.")
    raise

# Load the pre-trained model (assuming 'brain_tumor_model.h5' is present in your working directory)
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define the image size that the model expects
IMG_SIZE = 224

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize the image to the expected size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image pixels to [0, 1]
    return img_array

# Define the function to predict brain tumor
def predict_tumor(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Get the index of the highest prediction
    return class_idx, predictions[0][class_idx]  # Return the class index and confidence score

# Create the Streamlit interface
st.title('Brain Tumor Detection using X-ray Images')
st.write("Upload an X-ray image for analysis.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Perform prediction when the button is clicked
    if st.button("Detect Tumor"):
        st.write("Classifying the image...")

        # Make prediction
        class_idx, confidence = predict_tumor(img)

        # Show result based on class prediction
        if class_idx == 0:
            st.write(f"Result: No Tumor (Confidence: {confidence*100:.2f}%)")
        else:
            st.write(f"Result: Tumor Detected (Confidence: {confidence*100:.2f}%)")
