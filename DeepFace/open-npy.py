import numpy as np
# Load the .npy file
data = np.load('embeddings/Daniel.npy', allow_pickle=True).item()
# Print the loaded data
print(data)