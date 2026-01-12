import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --------------------
# Absolute base path
# --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_BASE = os.path.join(SCRIPT_DIR, "output")
MY_DIR = os.path.join(OUTPUT_BASE, "me")
RANDOM_DIR = os.path.join(OUTPUT_BASE, "random")

# --------------------
# Sanity checks
# --------------------
if not os.path.isdir(MY_DIR):
    raise RuntimeError(f"Missing folder: {MY_DIR}")

if not os.path.isdir(RANDOM_DIR):
    raise RuntimeError(f"Missing folder: {RANDOM_DIR}")

# --------------------
# Config
# --------------------
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --------------------
# Transform
# --------------------
transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# --------------------
# Dataset
# --------------------
class PairDataset(Dataset):
    def __init__(self, my_dir, random_dir, transform):
        self.my = [os.path.join(my_dir, f) for f in os.listdir(my_dir)]
        self.rand = [os.path.join(random_dir, f) for f in os.listdir(random_dir)]
        self.transform = transform

    def __len__(self):
        return max(len(self.my), len(self.rand))

    def __getitem__(self, idx):
        if random.random() < 0.5:
            img1 = Image.open(random.choice(self.my)).convert("RGB")
            img2 = Image.open(random.choice(self.my)).convert("RGB")
            label = 1.0
        else:
            img1 = Image.open(random.choice(self.my)).convert("RGB")
            img2 = Image.open(random.choice(self.rand)).convert("RGB")
            label = 0.0

        return (
            self.transform(img1),
            self.transform(img2),
            torch.tensor(label, dtype=torch.float32)
        )

dataset = PairDataset(MY_DIR, RANDOM_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------
# Model
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

# --------------------
# Loss
# --------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        dist = nn.functional.pairwise_distance(e1, e2)
        loss = label * dist.pow(2) + \
               (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()

criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------
# Training
# --------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for img1, img2, label in loader:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        emb1 = model(img1)
        emb2 = model(img2)
        loss = criterion(emb1, emb2, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "face_embedding_model.pth")
print("Model saved")
