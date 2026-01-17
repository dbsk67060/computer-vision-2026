from deepface import DeepFace
import cv2

IMG_PATH = "data\Magnus\IMG_0626.png"

def show_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Kunne ikke læse billede")
        return

    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="retinaface",
        enforce_detection=False,
        align=False
    )

    print("faces len:", len(faces))
    if not faces:
        print("Ingen ansigter fundet overhovedet")
        return

    face = faces[0]
    fa = face["facial_area"]
    print("facial_area:", fa)

    # Nogle versioner giver liste [x1,y1,x2,y2], andre dict med x,y,w,h
    if isinstance(fa, (list, tuple)):
        x1, y1, x2, y2 = fa
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
    else:
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

    # tegn boks
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # landmarks fra RetinaFace, hvis tilgængelige
    lm = fa.get("landmarks") if isinstance(fa, dict) else None
    if lm:
        for name, pt in lm.items():
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(
                img,
                name,
                (px + 2, py - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1
            )
    else:
        print("Ingen landmarks fundet i facial_area")

    # vis billedet i mindre størrelse
    h0, w0 = img.shape[:2]
    factor = min(1.0, 800 / max(h0, w0))  # maks bredde/højde ≈ 800 px
    img_small = cv2.resize(img, (int(w0 * factor), int(h0 * factor)))

    cv2.imshow("Landmarks demo", img_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_landmarks(IMG_PATH)
