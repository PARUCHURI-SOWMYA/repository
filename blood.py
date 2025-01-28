import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pre-trained model (replace with your actual model path)
model = tf.keras.models.load_model('path_to_your_trained_model')

def preprocess_image(image):
    """
    Preprocess the image for the model input.
    This includes resizing, normalizing, and reshaping.
    """
    try:
        # Resize image to match the model input size (e.g., 224x224)
        img = image.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Expand dimensions to simulate a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def predict_blood_group(image):
    """
    Predict the blood group using the trained model.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)

    if processed_image is None:
        return None

    # Make the prediction
    prediction = model.predict(processed_image)

    # Assuming the model's output is a one-hot encoded vector for blood groups
    # or an integer (e.g., 0: A, 1: B, 2: AB, 3: O)
    blood_group_map = {0: "A", 1: "B", 2: "AB", 3: "O"}  # Update based on your model's labels

    # Get the predicted blood group
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get index of the highest probability
    return blood_group_map.get(predicted_label, "Unknown")

def main():
    st.title("Blood Group Detection from Finger Image")

    # Upload the image
    uploaded_image = st.file_uploader("Upload a Finger Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the image using PIL
        image = Image.open(uploaded_image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the blood group using the pre-trained model
        blood_group = predict_blood_group(image)

        if blood_group:
            st.success(f"The predicted blood group is: {blood_group}")
        else:
            st.error("Error predicting the blood group.")
    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    main()
