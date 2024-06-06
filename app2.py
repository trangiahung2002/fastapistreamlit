import streamlit as st
from PIL import Image

st.title("Upload and Process Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to grayscale
    grayscale_image = image.convert('L')

    st.subheader("Processed Image (Grayscale)")
    st.image(grayscale_image, caption='Processed Image (Grayscale)', use_column_width=True)
