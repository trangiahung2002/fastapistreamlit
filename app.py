import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

st.title("Upload and Process Image")
st.markdown("[FastAPI Documentation](http://127.0.0.1:8000/docs)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    # st.image(image, caption='Uploaded Image', use_column_width=True)

    # Send image to FastAPI server
    response = requests.post(
        "http://127.0.0.1:8000/process_image/",
        files={"file": uploaded_file.getvalue()}
    )

    if response.status_code == 200:
        response_data = response.json()
        original_image_base64 = response_data["original_image"]
        processed_image_base64 = response_data["processed_image"]

        original_image = Image.open(BytesIO(base64.b64decode(original_image_base64)))
        processed_image = Image.open(BytesIO(base64.b64decode(processed_image_base64)))

        st.image(original_image, caption='Original Image (Base64)', use_column_width=True)
        st.subheader("Processed Image")
        st.image(processed_image, caption='Processed Image (Grayscale)', use_column_width=True)
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
