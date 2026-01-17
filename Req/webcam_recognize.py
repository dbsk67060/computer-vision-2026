# fast_webcam_opencv_arcface.py
import os
import cv2
import numpy as np
from deepface import DeepFace

EMB_DIR = "embeddings"
MODEL_NAME = "ArcFace"
NORMALIZATION = "ArcFace"
THRESHOLD = 0.68

# OpenCVs hurtige Haar-detektor
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def load_templates():
    templates = {}
    for file in os.listdir(EMB_DIR):
        if file.endswith("_mean.npy"):
            person = file.replace("_mean.npy", "")
            emb = np.load(os.path.join(EMB_DIR, file))
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            templates[person] = emb / norm
    return templates


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    templates = load_templates()
    if not templates:
        print("Ingen templates i embeddings/")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kunne ikke åbne webcam")
        return

    print("Tryk 'q' for at afslutte")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # nedskalér for fart
        h0, w0 = frame.shape[:2]
        target_w = 640
        scale = target_w / w0
        small = cv2.resize(frame, (target_w, int(h0 * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        label_text = "Ingen face"
        color = (0, 0, 255)

        if len(faces) > 0:
            # tag største face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

            # crop i small, skaler tilbage til original coords
            x0, y0 = int(x / scale), int(y / scale)
            w0, h0 = int(w / scale), int(h / scale)

            face_crop = frame[y0:y0 + h0, x0:x0 + w0]

            try:
                rep = DeepFace.represent(
                    img_path=face_crop,
                    model_name=MODEL_NAME,
                    detector_backend="skip",   # ingen ekstra detection
                    enforce_detection=False,
                    normalization=NORMALIZATION
                )
            except Exception:
                rep = []

            if rep:
                emb = np.array(rep[0]["embedding"], dtype=np.float32)
                if np.linalg.norm(emb) > 0:
                    emb = emb / np.linalg.norm(emb)

                    best_person = None
                    best_score = -1.0
                    for person, tmpl in templates.items():
                        score = cosine_similarity(emb, tmpl)
                        if score > best_score:
                            best_score = score
                            best_person = person

                    if best_score >= THRESHOLD:
                        label_text = f"{best_person} ({best_score:.2f})"
                        color = (0, 255, 0)
                    else:
                        label_text = f"Ukendt ({best_score:.2f})"

            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), color, 2)
            cv2.putText(frame, label_text, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Hurtig ArcFace + OpenCV", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
