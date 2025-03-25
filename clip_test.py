import clip
from PIL import Image
import torch

model, preprocess = clip.load("RN50")
model.cuda().eval()
image = Image.open("../../SSD/PiWiFi/NYCU/Env0/img/F1/1_posi/240506_172902/1714987812695449649_mask.png").convert("RGB")
image = preprocess(image)
image_input = torch.tensor(image).cuda()
image_input = image_input.unsqueeze(0)
with torch.no_grad():
    print(image_input.shape)
    image_features = model.encode_image(image_input).float()
    print(image_features.shape)