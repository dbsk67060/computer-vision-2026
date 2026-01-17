# preprocess_og_embed.py
# Udtrækker embeddings med DeepFace (ArcFace)
# Logger embeddings til TensorBoard (Projector)
# Ingen træning, ingen loss – kun evaluering

import os
import cv2
import numpy as np
from deepface import DeepFace

# TensorBoard (PyTorch)
from torch.utils.tensorboard import SummaryWriter
import torch

# =========================
# KONFIGURATION
# =========================
DATA_DIR = "data"
EMB_DIR = "embeddings"
TB_DIR = "runs/embeddings"

MODEL_NAME = "ArcFace"
NORMALIZATION = "ArcFace"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

# ÉN writer til hele scriptet
writer = SummaryWriter(TB_DIR)


# =========================
# PROCESSÉR ÉN PERSON
# =========================
def process_person(person_name):
    person_dir = os.path.join(DATA_DIR, person_name)
    files = [f for f in os.listdir(person_dir) if not f.startswith(".")]

    if not files:
        print(f"{person_name}: ingen billeder")
        return

    embeddings = []

    print(f"\n[{person_name}] behandler {len(files)} billeder")

    for file in files:
        path = os.path.join(person_dir, file)

        img = cv2.imread(path)
        if img is None:
            continue

        # Nedskalér store billeder (performance)
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > 800:
            scale = 800 / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        try:
            rep = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                normalization=NORMALIZATION
            )
        except Exception:
            continue

        if not rep:
            continue

        emb = np.array(rep[0]["embedding"], dtype=np.float32)
        embeddings.append(emb)

    if not embeddings:
        print(f"{person_name}: 0 embeddings")
        return

    # =========================
    # POST-PROCESSING
    # =========================
    embeddings = np.vstack(embeddings)

    mean_emb = embeddings.mean(axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    np.save(os.path.join(EMB_DIR, f"{person_name}.npy"), embeddings)
    np.save(os.path.join(EMB_DIR, f"{person_name}_mean.npy"), mean_emb)

    print(f"{person_name}: {embeddings.shape[0]} embeddings gemt")

    # =========================
    # TENSORBOARD LOGGING
    # =========================
    emb_tensor = torch.tensor(embeddings)
    labels = [person_name] * len(embeddings)

    writer.add_embedding(
        emb_tensor,
        metadata=labels,
        tag=f"{person_name}_embeddings"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)
        if os.path.isdir(person_path):
            process_person(person)



    writer.close()
    print("\nFærdig. Start TensorBoard med:")
    print("tensorboard --logdir runs")
