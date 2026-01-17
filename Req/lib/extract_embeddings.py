# extract_embeddings.py
import numpy as np
import cv2
import os
from deepface import DeepFace
from config import MODEL_NAME, ALIGNED_SIZE

ALIGNED_DIR = "aligned"
EMB_DIR = "embeddings"

os.makedirs(EMB_DIR, exist_ok=True)

for person in os.listdir(ALIGNED_DIR):
    person_dir = os.path.join(ALIGNED_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        # Resize for sikkerhed (skulle allerede være 112x112)
        img = cv2.resize(img, (ALIGNED_SIZE, ALIGNED_SIZE))

        try:
            result = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend="skip",    # Allerede aligned
                enforce_detection=False,
                normalization="ArcFace"     # Vigtigt for ArcFace
            )
        except Exception:
            continue

        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            embeddings.append(emb)

    if not embeddings:
        print(f"Ingen embeddings for {person}")
        continue

    embeddings = np.vstack(embeddings)
    mean_emb = embeddings.mean(axis=0)

    # Normaliser mean embedding (for cosine)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    np.save(os.path.join(EMB_DIR, f"{person}.npy"), embeddings)
    np.save(os.path.join(EMB_DIR, f"{person}_mean.npy"), mean_emb)

    print(f"{person}: {len(embeddings)} embeddings gemt")
