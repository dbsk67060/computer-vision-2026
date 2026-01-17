# preprocess.py
import cv2
import numpy as np
from deepface import DeepFace
import os
from config import DETECTOR_BACKEND, ALIGNED_SIZE, MIN_FACE_SIZE

# ArcFace reference landmarks (standard 112x112)
ARCFACE_REF = np.array([
    [38.2946, 51.6963],  # venstre øje
    [73.5318, 51.5014],  # højre øje
    [56.0252, 71.7366],  # næse
    [41.5493, 92.3655],  # venstre mundvig
    [70.7299, 92.2041],  # højre mundvig
], dtype=np.float32)


def align_face(img, landmarks, size=112):
    """Affine transformation til ArcFace-format."""
    src = np.array(landmarks, dtype=np.float32)
    dst = ARCFACE_REF * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    aligned = cv2.warpAffine(img, M, (size, size), borderValue=0.0)
    return aligned


def process_person(person_dir, output_dir):
    """Behandl alle billeder for én person."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        try:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=False  # Vi laver vores egen alignment
            )
        except Exception:
            continue

        if not faces:
            continue

        face = faces[0]
        region = face["facial_area"]

        # Tjek minimumstørrelse
        if region["w"] < MIN_FACE_SIZE or region["h"] < MIN_FACE_SIZE:
            continue

        # Hent landmarks (øjne, næse, mund)
        # RetinaFace returnerer landmarks i facial_area
        if "left_eye" in region and region["left_eye"]:
            landmarks = [
                region["left_eye"],
                region["right_eye"],
                region.get("nose", [(region["x"] + region["w"]//2), (region["y"] + region["h"]//2)]),
                region.get("mouth_left", region["left_eye"]),
                region.get("mouth_right", region["right_eye"]),
            ]
            aligned = align_face(img, landmarks, ALIGNED_SIZE)
        else:
            # Fallback: brug DeepFace's alignment
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cropped = img[y:y+h, x:x+w]
            aligned = cv2.resize(cropped, (ALIGNED_SIZE, ALIGNED_SIZE))

        out_path = os.path.join(output_dir, f"aligned_{count:03d}.jpg")
        cv2.imwrite(out_path, aligned)
        count += 1

    return count


if __name__ == "__main__":
    DATA_DIR = "data"
    ALIGNED_DIR = "aligned"

    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_path):
            continue

        out_path = os.path.join(ALIGNED_DIR, person)
        n = process_person(person_path, out_path)
        print(f"{person}: {n} aligned faces")
