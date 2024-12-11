import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
base_folder = r"C:\Users\royma\OneDrive\Desktop\FLIPCART GRID\PROJECTSTART" # give the path to csv and images
csv_file = os.path.join(base_folder, "brocoli.csv")

# Load labels from CSV
data = pd.read_csv(csv_file)

# Initialize data arrays
images = []
freshness = []
lifespan = []

# Loop through the dataset and preprocess images for Broccoli
for _, row in data.iterrows():
    if row["Category"] == "BROCOLI":  # Only process Broccoli
        img_path = os.path.join(base_folder, row["Category"], row["Image Name"])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0  # Normalize the image
            images.append(img)
            freshness.append(row["Freshness"])
            lifespan.append(row["Expected Lifespan"])

# Convert to NumPy arrays
images = np.array(images, dtype="float32")
freshness = np.array(freshness)
lifespan = np.array(lifespan)

# Split data into train and test sets
X_train, X_test, y_train_fresh, y_test_fresh = train_test_split(images, freshness, test_size=0.2, random_state=42)
_, _, y_train_life, y_test_life = train_test_split(images, lifespan, test_size=0.2, random_state=42)

# Save preprocessed data for Broccoli
np.savez("broccoli_preprocessed_data.npz",
         X_train=X_train, X_test=X_test,
         y_train_fresh=y_train_fresh, y_test_fresh=y_test_fresh,
         y_train_life=y_train_life, y_test_life=y_test_life)

print("Broccoli data preprocessing completed. Saved to 'broccoli_preprocessed_data.npz'.")
