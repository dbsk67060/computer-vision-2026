import numpy as np
import os

EXCLUDE = {"Others"}

def load_all_embeddings(folder="embeddings"):
    database = {}

    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue

        name = file.replace(".npy", "")

        if name in EXCLUDE:
            continue   # ← KRITISK

        database[name] = np.load(os.path.join(folder, file))

    return database
