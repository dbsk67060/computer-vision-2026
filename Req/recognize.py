# recognize.py
import numpy as np
import cv2
import os
from deepface import DeepFace
from config import MODEL_NAME, DETECTOR_BACKEND, THRESHOLD, ALIGNED_SIZE

EMB_DIR = "Req/embeddings"


def load_templates():
    """Indlæs alle mean embeddings."""
    templates = {}
    for file in os.listdir(EMB_DIR):
        if file.endswith("_mean.npy"):
            person = file.replace("_mean.npy", "")
            templates[person] = np.load(os.path.join(EMB_DIR, file))
    return templates


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize(img_path, templates):
    """Genkend person i billede."""
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            normalization="ArcFace"
        )
    except Exception as e:
        return None, 0.0

    if not result:
        return None, 0.0

    emb = np.array(result[0]["embedding"], dtype=np.float32)
    emb = emb / np.linalg.norm(emb)

    best_match = None
    best_score = -1

    for person, template in templates.items():
        score = cosine_similarity(emb, template)
        if score > best_score:
            best_score = score
            best_match = person

    if best_score >= THRESHOLD:
        return best_match, best_score
    return None, best_score


if __name__ == "__main__":
    templates = load_templates()
    test_img = "test.jpg"  # Dit testbillede

    person, score = recognize(test_img, templates)
    if person:
        print(f"Genkendt: {person} (score: {score:.3f})")
    else:
        print(f"Ukendt person (bedste score: {score:.3f})")
