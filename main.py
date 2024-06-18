from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image
import uvicorn
from ultralytics import YOLO
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
import numpy as np
import cv2

app = FastAPI()

model_biomed, biomed_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomed_tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model_dict = {
    'brain': YOLO('weights/brain.pt'),
    'chest': YOLO('weights/chest.pt'),
    'breast_ultrasound_detect': YOLO('weights/breast-ultrasound-detect.pt'),
    'breast_ultrasound_cls': YOLO('weights/breast-ultrasound-cls.pt'),
    'breast_mammo_detect': YOLO('weights/breast-mammo-detect.pt'),
    'breast_mammo_cls_YesNo': YOLO('weights/breast-mammo-cls-YesNo.pt'),
    'breast_mammo_cls_BenignMalignant': YOLO('weights/breast-mammo-cls-BenignMalignant.pt'),
    'kidney': YOLO('weights/kidney-stone.pt')
}

# BiomedCLIP
def biomed_classify(image, labels):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_biomed.to(device)
    model_biomed.eval()

    context_length = 256
    image_tensor = biomed_preprocess(image).unsqueeze(0).to(device)
    texts = biomed_tokenizer([f'this is {l}' for l in labels], context_length=context_length).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model_biomed(image_tensor, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        logits = logits.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

    top_label_index = sorted_indices[0][0]
    top_label = labels[top_label_index]

    return top_label

def convert_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def getDetectionInfo(image, model, confidence_threshold):
    image_array = np.array(image.convert("RGB"))

    results = model.predict(source=image, save=False, conf=confidence_threshold / 100)

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

    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        label = f"{obj['label']} ({obj['confidence'] * 100:.2f}%)"  # Convert confidence to percentage

        box_color = (255, 255, 0)  # RGB format for yellow color
        text_color = (0, 0, 0)  # Black text color

        cv2.rectangle(image_array, (x1, y1), (x2, y2), box_color, 2)
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image_array, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), box_color,
                              -1)
        cv2.putText(image_array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                            font_thickness)
    return Image.fromarray(image_array)


@app.post("/predict_images_bodypart/")
async def process_images(files: list[UploadFile] = File(...), bodypart: str = 'brain', confidence_threshold: int = Query(50, ge=0, le=100)):
    if bodypart not in ['brain', 'chest', 'breast', 'kidney']:
        raise HTTPException(status_code=400, detail="Invalid body part specified")

    try:
        results = []
        for file in files:
            img = Image.open(file.file)

            if bodypart == 'breast':
                breast_labels = ['ultrasound', 'mammography']
                breast_label = biomed_classify(img, breast_labels)
                if breast_label == 'ultrasound':
                    model = model_dict['breast_ultrasound_detect']
                else:
                    model = model_dict['breast_mammo_detect']
            else:
                model = model_dict[bodypart]

            plotted_image = getDetectionInfo(img, model, confidence_threshold)
            plotted_image_base64 = convert_image_to_base64(plotted_image)
            results.append({
                "filename": file.filename,
                "plotted_image_base64": plotted_image_base64,
            })

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_images_type/")
async def process_images_type(files: list[UploadFile] = File(...), confidence_threshold: int = Query(50, ge=0, le=100), image_type: str = Query(...), body_part: str = None):
    try:
        results = []
        for file in files:
            img = Image.open(file.file)

            if image_type == 'MRI':
                model = model_dict['brain']
            elif image_type == 'X-ray':
                model = model_dict['chest']
            elif image_type == 'Ultrasound':
                # Sử dụng body_part từ request nếu có
                if body_part == 'breast':
                    model = model_dict['breast_ultrasound_detect']
                elif body_part == 'kidney':
                    model = model_dict['kidney']
                else:
                    raise HTTPException(status_code=400, detail="Invalid body part for ultrasound")
            elif image_type == 'Mammography':
                model = model_dict['breast_mammo_detect']
            else:
                raise HTTPException(status_code=400, detail="Invalid image type")

            plotted_image = getDetectionInfo(img, model, confidence_threshold)
            plotted_image_base64 = convert_image_to_base64(plotted_image)
            results.append({
                "filename": file.filename,
                "plotted_image_base64": plotted_image_base64,
            })

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)