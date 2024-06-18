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
import torch

hostname = 'http://localhost:8000'
# hostname = 'https://fastapistreamlit-im4zw4v7vq-et.a.run.app'

sample_brain_images = {
    "Sample 1": "samples/brain_sample (1).jpg",
    "Sample 2": "samples/brain_sample (2).jpg",
    "Sample 3": "samples/brain_sample (3).jpg",
    "Sample 4": "samples/brain_sample (4).jpg",
    "Sample 5": "samples/brain_sample (5).jpg",
}

sample_chest_images = {
    "Sample 1": "samples/chest_sample (1).jpg",
    "Sample 2": "samples/chest_sample (2).jpg",
    "Sample 3": "samples/chest_sample (3).jpg",
    "Sample 4": "samples/chest_sample (4).jpg",
    "Sample 5": "samples/chest_sample (5).jpg",
}

sample_breast_images = {
    "Sample 1": "samples/breast_sample (1).png",
    "Sample 2": "samples/breast_sample (2).png",
    "Sample 3": "samples/breast_sample (3).png",
    "Sample 4": "samples/breast_sample (4).jpg",
    "Sample 5": "samples/breast_sample (5).jpg",
}

sample_kidney_images = {
    "Sample 1": "samples/kidney_sample (1).jpg",
    "Sample 2": "samples/kidney_sample (2).jpg",
    "Sample 3": "samples/kidney_sample (3).jpg",
    "Sample 4": "samples/kidney_sample (4).jpg",
    "Sample 5": "samples/kidney_sample (5).jpg",
}

# print(torch.cuda.is_available())


def display_samples(sample_images):
    cols = st.columns(len(sample_images))
    for i, (sample_name, sample_path) in enumerate(sample_images.items()):
        with cols[i]:
            image = Image.open(sample_path)
            st.image(image, caption=sample_name, use_column_width=True)


# def draw_boxes(image, objects):
#     image_array = np.array(image.convert("RGB"))  # Convert to RGB to ensure correct color format
#     for obj in objects:
#         x1, y1, x2, y2 = map(int, obj['bbox'])
#         label = f"{obj['label']} ({obj['confidence'] * 100:.2f}%)"  # Convert confidence to percentage
#
#         # Change the color to yellow (RGB format)
#         box_color = (255, 255, 0)  # RGB format for yellow color
#         text_color = (0, 0, 0)  # Black text color
#
#         cv2.rectangle(image_array, (x1, y1), (x2, y2), box_color, 2)
#         font_scale = 0.5
#         font_thickness = 1
#         text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
#         text_x = x1
#         text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#         cv2.rectangle(image_array, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), box_color,
#                       -1)
#         cv2.putText(image_array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
#                     font_thickness)
#     return Image.fromarray(image_array)
#
#
# def runDetection(bodypart):
#     uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
#
#     if uploaded_files:
#
#         confidence_threshold = st.slider("Set the minimum possibility (%) of prediction", 0, 100, 50, step=10)  # Adjust slider to 0-100%
#
#         files = [("files", (file.name, file, file.type)) for file in uploaded_files]
#         response = requests.post(f'{hostname}/predict_images/', params={"bodypart": bodypart}, files=files)
#
#         if response.status_code == 200:
#             response_data = response.json()
#             results = response_data["results"]
#
#             for result in results:
#                 filename = result["filename"]
#                 processed_image_base64 = result["processed_image_base64"]
#                 detection_info = result["detection_info"]
#
#                 # Decode processed image from base64
#                 processed_image = Image.open(BytesIO(base64.b64decode(processed_image_base64)))
#
#                 # filtered_objects = [
#                 #     obj for obj in detection_info['objects']
#                 #     if obj['confidence'] >= confidence_threshold / 100  # Convert slider value back to decimal
#                 # ]
#
#                 # plotted_image = draw_boxes(processed_image, filtered_objects)
#
#                 st.subheader(f"Results for {filename}")
#                 col1, col2 = st.columns(2)
#
#                 with col1:
#                     st.image(processed_image, caption=f'Uploaded Image - {filename}', use_column_width=True)
#                 with col2:
#                     st.image(plotted_image, caption=f'Predicted Image - {filename}', use_column_width=True)
#                     result['detection_info']['objects'] = filtered_objects
#                 with st.expander("Show Info"):
#                     result['detection_info']['objects'] = filtered_objects
#                     st.json(result)
#         else:
#             st.error(f"Error: {response.status_code}, {response.text}")
def runDetection(bodypart):
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f'{bodypart}_uploaded_files')

    if uploaded_files:
        confidence_threshold = st.slider("Set the minimum possibility (%) of prediction", 0, 100, 50, step=10, key=f'{bodypart}_confidence_threshold')  # Adjust slider to 0-100%

        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        response = requests.post(f'{hostname}/predict_images_bodypart/', params={"bodypart": bodypart, "confidence_threshold": confidence_threshold}, files=files)

        if response.status_code == 200:
            response_data = response.json()
            results = response_data["results"]

            for i, result in enumerate(results):
                filename = result["filename"]
                plotted_image_base64 = result["plotted_image_base64"]

                # Decode plotted image from base64
                plotted_image = Image.open(BytesIO(base64.b64decode(plotted_image_base64)))

                st.subheader(f"Results for {filename}")
                col1, col2 = st.columns(2)

                with col1:
                    original_image = Image.open(uploaded_files[i])
                    st.image(original_image, caption=f'Uploaded Image - {filename}', use_column_width=True)
                with col2:
                    st.image(plotted_image, caption=f'Predicted Image - {filename}', use_column_width=True)
                with st.expander("Show Info"):
                    st.json(result)
        else:
            st.error(f"Error: {response.status_code}, {response.text}")

def runDetection2(image_type):
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f'{image_type}_uploaded_files')

    if uploaded_files:
        confidence_threshold = st.slider("Set the minimum possibility (%) of prediction", 0, 100, 50, step=10, key=f'{image_type}_confidence_threshold')  # Adjust slider to 0-100%

        files = [("files", (file.name, file, file.type)) for file in uploaded_files]

        # Lưu lựa chọn bộ phận cho từng ảnh
        body_parts = []
        for i, file in enumerate(uploaded_files):
            if image_type == "Ultrasound":
                body_part = st.selectbox(f"Select the body part for ultrasound ({file.name}):", ["breast", "kidney"], key=f'{image_type}_body_part_{i}')
                body_parts.append(body_part)
            else:
                body_parts.append(None)

        params = {"confidence_threshold": confidence_threshold, "image_type": image_type}
        response_data = []

        for i, file in enumerate(uploaded_files):
            file_params = params.copy()
            if body_parts[i]:
                file_params["body_part"] = body_parts[i]

            response = requests.post(f'{hostname}/predict_images_type/', params=file_params, files=[("files", (file.name, file, file.type))])
            if response.status_code == 200:
                response_data.append(response.json())
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
                return

        for i, response in enumerate(response_data):
            results = response["results"]
            for result in results:
                filename = result["filename"]
                plotted_image_base64 = result["plotted_image_base64"]

                # Decode plotted image from base64
                plotted_image = Image.open(BytesIO(base64.b64decode(plotted_image_base64)))

                st.subheader(f"Results for {filename}")
                col1, col2 = st.columns(2)

                with col1:
                    original_image = Image.open(uploaded_files[i])
                    st.image(original_image, caption=f'Uploaded Image - {filename}', use_column_width=True)
                with col2:
                    st.image(plotted_image, caption=f'Predicted Image - {filename}', use_column_width=True)
                with st.expander("Show Info"):
                    st.json(result)



def main():
    st.title("Main Page")
    st.write("Choose an option from the sidebar to proceed.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to",
                            ["Main Page", "Brain Tumor", "Chest Nodule", "Breast Cancer",
                             "Kidney Stone", "MRI Scans", "Xray Images", "Ultrasound Images", "Mammography"])
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = page
if st.session_state['current_page'] != page:
    st.session_state.clear()
    st.session_state['current_page'] = page

# Display the selected page
if page == "Main Page":
    main()
elif page == "Brain Tumor":
    st.title("Brain Tumor Detection")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    display_samples(sample_brain_images)
    runDetection('brain')
elif page == "Chest Nodule":
    st.title("Chest Nodule Detection")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    display_samples(sample_chest_images)
    runDetection('chest')
elif page == "Breast Cancer":
    st.title("Breast Cancer Detection")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    display_samples(sample_breast_images)
    runDetection('breast')
elif page == "Kidney Stone ":
    st.title("Kidney Stone Detection")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    display_samples(sample_kidney_images)
    runDetection('kidney')
elif page == "MRI Scans":
    st.title("MRI")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    runDetection2('MRI')
elif page == "Xray Images":
    st.title("Xray Images")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    runDetection2('X-ray')
elif page == "Ultrasound Images":
    st.title("Ultrasound Images")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    runDetection2('Ultrasound')
elif page == "Mammography":
    st.title("Mammography")
    st.markdown(f"[FastAPI Documentation]({hostname}/docs)")
    runDetection2('Mammography')
