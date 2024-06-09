# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import uvicorn
# from ultralytics import YOLO
#
# app = FastAPI()
# model_brain = YOLO('brain.pt')
# model_chest = YOLO('chest.pt')
#
# @app.post("/process_brain_image/")
# async def process_brain_image(file: UploadFile = File(...)):
#     try:
#         img = Image.open(file.file)
#
#         predicted_img, detection_info = getDetectionInfo(img, model_brain)
#
#         img_array = np.array(predicted_img)
#         processed_pil_img = Image.fromarray(img_array)
#
#         # original_base64 = convert_image_to_base64(img)
#         processed_base64 = convert_image_to_base64(processed_pil_img)
#
#         return JSONResponse(content={
#             "detection_info": detection_info,
#             # "original_image_base64": original_base64,
#             "processed_image_base64": processed_base64,
#         })
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#
# @app.post("/process_chest_image/")
# async def process_chest_image(file: UploadFile = File(...)):
#     try:
#         img = Image.open(file.file)
#
#         predicted_img, detection_info = getDetectionInfo(img, model_chest)
#
#         img_array = np.array(predicted_img)
#         processed_pil_img = Image.fromarray(img_array)
#
#         # original_base64 = convert_image_to_base64(img)
#         processed_base64 = convert_image_to_base64(processed_pil_img)
#
#         return JSONResponse(content={
#             "detection_info": detection_info,
#             # "original_image_base64": original_base64,
#             "processed_image_base64": processed_base64,
#         })
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#
# def convert_image_to_base64(image: Image.Image) -> str:
#     buffered = BytesIO()
#     image.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode()
#
# def getDetectionInfo(image, model):
#
#     results = model.predict(source=image, save=False)
#
#     # for result in results:
#     #     plotted_image = result.plot()
#
#     width, height = results[0].orig_shape
#
#     objects = []
#     for box in results[0].boxes:
#         label = model.names[int(box.cls[0])]
#         confidence = float(box.conf[0])
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#
#         obj = {
#             "label": label,
#             "bbox": [x1, y1, x2, y2],
#             "confidence": confidence
#         }
#         objects.append(obj)
#
#     processing_time = results[0].speed
#
#     data = {
#         "size": {
#             "width": width,
#             "height": height
#         },
#         "objects": objects,
#         "processing_time": {
#             "preprocess": processing_time['preprocess'],
#             "inference": processing_time['inference'],
#             "postprocess": processing_time['postprocess']
#         }
#     }
#     return image, data
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from ultralytics import YOLO

app = FastAPI()
model_brain = YOLO('weights/brain.pt')
model_chest = YOLO('weights/chest.pt')

def convert_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def getDetectionInfo(image, model):
    results = model.predict(source=image, save=False)
    width, height = results[0].orig_shape

    objects = []
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        obj = {
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence
        }
        objects.append(obj)

    processing_time = results[0].speed

    data = {
        "size": {
            "width": width,
            "height": height
        },
        "objects": objects,
        "processing_time": {
            "preprocess": processing_time['preprocess'],
            "inference": processing_time['inference'],
            "postprocess": processing_time['postprocess']
        }
    }
    return image, data

@app.post("/predict_brain_images/")
async def process_brain_images(files: list[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            img = Image.open(file.file)
            predicted_img, detection_info = getDetectionInfo(img, model_brain)
            processed_base64 = convert_image_to_base64(predicted_img)
            results.append({
                "filename": file.filename,
                "detection_info": detection_info,
                "processed_image_base64": processed_base64,
            })

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predicts_chest_images/")
async def process_chest_images(files: list[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            img = Image.open(file.file)
            predicted_img, detection_info = getDetectionInfo(img, model_chest)
            processed_base64 = convert_image_to_base64(predicted_img)
            results.append({
                "filename": file.filename,
                "detection_info": detection_info,
                "processed_image_base64": processed_base64,
            })

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
