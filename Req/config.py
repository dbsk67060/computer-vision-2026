# config.py
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"   # Bedste balance mellem nøjagtighed og hastighed
ALIGNED_SIZE = 112                # ArcFace forventer 112x112
MIN_FACE_SIZE = 40                # Ignorer meget små ansigter
DISTANCE_METRIC = "cosine"        # Cosine virker godt med ArcFace
THRESHOLD = 0.68                  # Typisk threshold for ArcFace cosine
