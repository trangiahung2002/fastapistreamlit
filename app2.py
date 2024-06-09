from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model_brain = YOLO('brain.pt')

img = 'img3.jpg'
results = model_brain.predict(source=img, save=True)

for result in results:
    res_plotted = result.plot()



# cv2.imshow('YOLOv8 Detection', res_plotted)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

filename = results[0].path  # Đường dẫn đến ảnh
width, height = results[0].orig_shape  # Kích thước ảnh gốc
format = filename.split('.')[-1]  # Định dạng file ảnh

print("Tên file:", filename)
print("Kích thước:", width, "x", height)
print("Định dạng:", format)

objects = []
for box in results[0].boxes:
    label = model_brain.names[int(box.cls[0])]  # Nhãn của đối tượng
    confidence = float(box.conf[0])  # Độ tin cậy
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Tọa độ bounding box

    obj = {
        "label": label,
        "bbox": [x1, y1, x2, y2],
        "confidence": confidence
    }
    objects.append(obj)

print("Số lượng đối tượng:", len(objects))
print("Danh sách đối tượng:", objects)

processing_time = results[0].speed  # Thời gian xử lý (tính bằng giây)
print("Thời gian xử lý:", processing_time, "giây")