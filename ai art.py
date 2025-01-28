import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Set the DeepAI API key here
API_KEY = 'your_deepai_api_key_here'

# Function to generate artwork from text
def generate_art(text_description):
    url = 'https://api.deepai.org/api/text2img'
    headers = {'api-key': API_KEY}
    data = {'text': text_description}
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        image_url = response.json()['output_url']
        return image_url
    else:
        st.error("Error generating art.")
        return None

# Streamlit UI
st.title("AI Art Generator")
st.write("Enter a description, and let the AI create some art for you!")

text_input = st.text_area("Describe the art you want to create:")

if st.button('Generate Art') and text_input:
    with st.spinner('Generating your artwork...'):
        image_url = generate_art(text_input)
        
        if image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Generated Art", use_column_width=True)
        else:
            st.error("Failed to generate art. Please try again.")

