import torch
import os
import gdown
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_0, SqueezeNet
from torch.serialization import add_safe_globals
from torch.nn import Sequential
# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def transform_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)

def load_model(model_path="model/brain_tumor_squeezenet.pth"):
    if not os.path.exists(model_path):
        file_id = "1mIDGIewD4nXiVHBH1xgbzL54LYjgdu2l"
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)

    # 👉 Allow loading of SqueezeNet class
    from torch.serialization import add_safe_globals
    from torchvision.models.squeezenet import SqueezeNet
    
    # 🔁 Load full model object (not just weights)
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model


def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)  # ✅ Here model must be a full model
        _, predicted = torch.max(outputs, 1)

    class_names = ["1", "2", "3", "4"]
    return class_names[predicted.item()]
