from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import uvicorn

app = FastAPI()

def convert_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        img_array = np.array(img)

        # Process image (convert to grayscale)
        img = img.convert("RGB")
        processed_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        processed_pil_img = Image.fromarray(processed_img)

        original_base64 = convert_image_to_base64(img)
        processed_base64 = convert_image_to_base64(processed_pil_img)

        return JSONResponse(content={
            "original_image": original_base64,
            "processed_image": processed_base64
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
