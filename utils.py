import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_def import BrainTumorCNN
import gdown

MODEL_PATH = 'model/brain_tumor_model.pth'
GDRIVE_FILE_ID = '12MjvuVkhOd_DiCSircRRveH-9v2W6a0D'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'

def download_model():
    os.makedirs('model', exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists locally.")

def load_model(path=MODEL_PATH):
    download_model()  # Ensure model is available
    model = BrainTumorCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return "Tumor" if predicted.item() == 1 else "No Tumor"
