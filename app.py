import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

st.title("Upload and Process Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.markdown("[FastAPI Documentation](http://localhost:8000/docs)")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Send image to FastAPI server
    response = requests.post("http://localhost:8000/process_image", json={"data": img_str})

    # Display processed image
    if response.status_code == 200:
        processed_image = Image.open(BytesIO(base64.b64decode(response.json()["processed_image"])))
        st.image(processed_image, caption='Processed Image', use_column_width=True)