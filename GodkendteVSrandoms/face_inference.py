import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# --------------------
# Paths
# --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, "face_embedding_model.pth")
REFERENCE_DIR = os.path.join(SCRIPT_DIR, "output", "me")     # kendte billeder af dig
TEST_DIR = os.path.join(SCRIPT_DIR, "test_images")           # nye billeder

# --------------------
# Config
# --------------------
IMG_SIZE = 160
THRESHOLD = 0.75   # justeres empirisk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --------------------
# Transform (SAMME som træning – uden augmentation)
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# --------------------
# Model (SAMME SOM TRÆNING)
# --------------------
class FaceEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

model = FaceEmbeddingNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------------
# Build reference embedding (DIG)
# --------------------
def build_reference_embedding(folder):
    embeddings = []

    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(img)
            embeddings.append(emb.cpu())

    if not embeddings:
        raise RuntimeError("No reference images found")

    ref = torch.mean(torch.cat(embeddings), dim=0)
    return ref

reference_embedding = build_reference_embedding(REFERENCE_DIR)

# --------------------
# Test images
# --------------------
for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img).cpu()

    distance = torch.norm(emb - reference_embedding).item()

    if distance < THRESHOLD:
        result = "GODKENDT"
    else:
        result = "RANDOM"

    print(f"{fname:25} → {result:8}  dist={distance:.3f}")
