import streamlit as st
from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    Preprocess the image: resize and convert it to grayscale for further processing.
    """
    try:
        # Resize image to match a consistent size (224x224 for example)
        img = image.resize((224, 224))

        # Convert to numpy array for further processing
        img_array = np.array(img)

        # Convert to grayscale
        gray_img = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale manually

        # Return processed image
        return gray_img
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def main():
    st.title("Blood Group Detection from Finger Image")

    # Upload the image
    uploaded_image = st.file_uploader("Upload a Finger Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the image using PIL
        image = Image.open(uploaded_image)

        # Display the original image using Streamlit's built-in image display
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        result_image = preprocess_image(image)

        if result_image is not None:
            # Image processed, print success message
            st.success("Image processed successfully.")

            # Display the processed image using Streamlit's built-in image display
            st.image(result_image, caption="Processed Grayscale Image", use_column_width=True, clamp=True)

            # Optionally, you can save the processed image
            st.write("Processed grayscale image is ready.")
        else:
            st.error("Error processing the image.")
    else:
        st.info("Please upload an image.")

if __name__ == "__main__":
    main()
