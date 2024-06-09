# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO
# import base64
# import cv2
# import numpy as np
#
# def draw_boxes(image, objects):
#     image_array = np.array(image)
#     for obj in objects:
#         x1, y1, x2, y2 = map(int, obj['bbox'])
#         label = f"{obj['label']} ({obj['confidence']:.2f})"
#
#         cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         font_scale = 0.5
#         font_thickness = 1
#         text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#         text_x = x1
#         text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#
#         cv2.rectangle(image_array, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5),
#                       (0, 255, 0), -1)
#
#         cv2.putText(image_array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
#                     font_thickness)
#
#     return Image.fromarray(image_array)
# def main():
#     st.title("Main Page")
#     st.write("Choose an option from the sidebar to proceed.")
#
# def runDetection(apilink):
#     # st.markdown("[FastAPI Documentation](https://fastapistreamlit-im4zw4v7vq-et.a.run.app/docs)")
#     st.markdown("[FastAPI Documentation](http://localhost:8000/docs)")
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.1)
#
#         image = Image.open(uploaded_file)
#
#         # Send image to FastAPI server
#         response = requests.post(
#             apilink,
#             files={"file": uploaded_file.getvalue()}
#         )
#
#         if response.status_code == 200:
#             response_data = response.json()
#             processed_image_base64 = response_data["processed_image_base64"]
#             detection_info = response_data["detection_info"]
#
#             # Decode processed image from base64
#             processed_image = Image.open(BytesIO(base64.b64decode(processed_image_base64)))
#
#             filtered_objects = [
#                 obj for obj in detection_info['objects']
#                 if obj['confidence'] >= confidence_threshold
#             ]
#
#             plotted_image = draw_boxes(processed_image, filtered_objects)
#
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.image(image, caption='Uploaded Image', use_column_width=True)
#             with col2:
#                 st.image(plotted_image, caption='Predicted Image', use_column_width=True)
#                 response_data['detection_info']['objects'] = filtered_objects
#                 st.json(response_data)
#
#         else:
#             st.error(f"Error: {response.status_code}, {response.text}")
#
#
#
# # Sidebar for navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.selectbox("Go to", ["Main Page", "Brain Tumor Detection", "Chest Nodule Detection"])
#
# # Display the selected page
# if page == "Main Page":
#     main()
# elif page == "Brain Tumor Detection":
#     st.title("Brain Tumor Detection")
#     runDetection('http://localhost:8000/process_brain_image/')
# elif page == "Chest Nodule Detection":
#     st.title("Chest Nodule Detection")
#     runDetection('http://localhost:8000/process_chest_image/')

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np

hostname = 'http://localhost:8000'

def draw_boxes(image, objects):
    image_array = np.array(image)
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        label = f"{obj['label']} ({obj['confidence']:.2f})"
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image_array, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5),
                      (0, 255, 0), -1)
        cv2.putText(image_array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thickness)
    return Image.fromarray(image_array)

def runDetection(apilink):

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")

    if uploaded_files:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, step=0.1)

        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        response = requests.post(apilink, files=files)

        if response.status_code == 200:
            response_data = response.json()
            results = response_data["results"]

            for result in results:
                filename = result["filename"]
                processed_image_base64 = result["processed_image_base64"]
                detection_info = result["detection_info"]

                # Decode processed image from base64
                processed_image = Image.open(BytesIO(base64.b64decode(processed_image_base64)))

                filtered_objects = [
                    obj for obj in detection_info['objects']
                    if obj['confidence'] >= confidence_threshold
                ]

                plotted_image = draw_boxes(processed_image, filtered_objects)

                st.subheader(f"Results for {filename}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(processed_image, caption=f'Uploaded Image - {filename}', use_column_width=True)
                with col2:
                    st.image(plotted_image, caption=f'Predicted Image - {filename}', use_column_width=True)
                    result['detection_info']['objects'] = filtered_objects
                with st.expander("Show Info"):
                    result['detection_info']['objects'] = filtered_objects
                    st.json(result)
        else:
            st.error(f"Error: {response.status_code}, {response.text}")

def main():
    st.title("Main Page")
    st.write("Choose an option from the sidebar to proceed.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Main Page", "Brain Tumor Detection", "Chest Nodule Detection"])

# Display the selected page
if page == "Main Page":
    main()
elif page == "Brain Tumor Detection":
    st.title("Brain Tumor Detection")
    runDetection(f'{hostname}/process_brain_images/')
elif page == "Chest Nodule Detection":
    st.title("Chest Nodule Detection")
    runDetection(f'{hostname}/process_chest_images/')


