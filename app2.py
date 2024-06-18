import torch
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

# Tạo mô hình và tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Đường dẫn đến ảnh MRI não của bạn
brain_mri_image_path = 'mammo (1).png'  # Thay thế bằng đường dẫn tệp cục bộ của bạn

# Mở ảnh MRI não
image = Image.open(brain_mri_image_path)

# Chuẩn bị nhãn và văn bản
template = 'this is '
labels = [
    'ultrasound',
    'mammography'
]

# Thiết bị
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Chuẩn bị ảnh và văn bản
context_length = 256
image_tensor = preprocess(image).unsqueeze(0).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)

# Dự đoán
with torch.no_grad():
    image_features, text_features, logit_scale = model(image_tensor, texts)
    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

# Lấy nhãn xếp đầu tiên
top_label_index = sorted_indices[0][0]
top_label = labels[top_label_index]
# top_prob = logits[0][top_label_index] * 100

# In nhãn xếp đầu tiên và xác suất tương ứng
print(top_label)

model_yolo_mammo = YOLO('weights/breast-mammo-cls-YesNo.pt')
model_yolo_ulso = YOLO('weights/breast-ultrasound-cls.pt')
model_predict = YOLO('weights/breast-ultrasound-detect.pt')

# model_yolo_ulso.predict('ultra (1).jpg')
cv2_image = cv2.imread('ultra (1).jpg')
cv2_image = np.array(cv2_image)
# pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

result = model_predict.predict(source=cv2_image, device=device, conf=0.5)

# conf = result[0].probs.data.tolist()
# print(conf)
# #
# if (conf[0] > conf[1]): #with helmet
#     cs = conf[0] - conf[1]
#     if cs > conf_cls_set:s
#         # return [True, cs]
#     else:
#         # return [None, 0]
# else:
#     cs = conf[1] - conf[0]
#     if cs > conf_cls_set:
#         # return [False, cs] #withoutHelmet
#     else:
#         # return [None, 0]