from deepface import DeepFace
import numpy as np
import os
import cv2

DATA_DIR = "data"
OUT_DIR = "embeddings"
CROP_DIR = "cropped_faces\Karla"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

MODEL_NAME = "ArcFace"  # eller "Facenet512"
DETECTOR = "retinaface"

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Behandler {person}")
    embeddings = []

    person_crop_dir = os.path.join(CROP_DIR, person)
    os.makedirs(person_crop_dir, exist_ok=True)

    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)

        # Load billede
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=DETECTOR,
                enforce_detection=True,   # behold, så vi ikke får tomme / random crops
                align=True                # sikrer bedre alignment
            )
        except Exception:
            # Skip hvis detection fejler
            continue

        if len(faces) == 0:
            continue

        # Brug første ansigt (forudsat 1 person pr. billede)
        face = faces[0]
        region = face["facial_area"]

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cropped = img[y:y+h, x:x+w]

        # (valgfrit) filtrér meget små ansigter
        if w < 60 or h < 60:
            continue

        crop_name = f"crop_{file}"
        cv2.imwrite(os.path.join(person_crop_dir, crop_name), cropped)

        # Repræsentation på cropped face
        try:
            result = DeepFace.represent(
                img_path=cropped,
                model_name=MODEL_NAME,
                detector_backend="skip",   # vigtigt, da vi allerede har cropped
                enforce_detection=False
            )
        except Exception:
            continue

        if len(result) == 0:
            continue

        emb = np.array(result[0]["embedding"], dtype="float32")
        embeddings.append(emb)

    if len(embeddings) == 0:
        print(f"Ingen embeddings for {person}")
        continue

    embeddings = np.vstack(embeddings)

    # (vigtigt) lav også en gennemsnitsvektor til brug som “template”
    mean_emb = embeddings.mean(axis=0)

    np.save(os.path.join(OUT_DIR, f"{person}.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, f"{person}_mean.npy"), mean_emb)

    print(f"  gemt {embeddings.shape[0]} embeddings for {person}")
