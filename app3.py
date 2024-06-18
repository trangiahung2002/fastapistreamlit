from ultralytics import YOLO

import cv2

model = YOLO('weights/brain.pt')

image = 'samples/brain_sample (1).jpg'

results = model.predict(image, save=False, conf=0.5)

for result in results:
    # Vẽ bounding box và nhãn lên ảnh
    res_plotted = result.plot()

    # Chuyển ảnh về định dạng BGR (OpenCV)
    res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

    # Hiển thị ảnh
    cv2.imshow("YOLOv8 Detection", res_plotted)
    cv2.waitKey(0)

cv2.destroyAllWindows()