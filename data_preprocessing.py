import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
base_folder = r"C:\Users\royma\OneDrive\Desktop\FLIPCART GRID\PROJECTSTART" # give the path to csv and images
csv_file = os.path.join(base_folder, "labels.csv")

# Load labels from CSV
data = pd.read_csv(csv_file)

# Initialize data arrays
images = []
categories = []
freshness = []
lifespan = []

# Category mapping
category_map = {"BROCOLI": 0, "ONION": 1, "PAPAYA": 2}

# Loop through the dataset and preprocess images
for _, row in data.iterrows():
    img_path = os.path.join(base_folder, row["Category"], row["Image Name"])
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0  # Normalize the image
        images.append(img)
        categories.append(category_map[row["Category"]])
        freshness.append(row["Freshness"])
        lifespan.append(row["Expected Lifespan"])

# Convert to NumPy arrays
images = np.array(images, dtype="float32")
categories = np.array(categories)
freshness = np.array(freshness)
lifespan = np.array(lifespan)

# One-hot encode categories
categories_encoded = to_categorical(categories, num_classes=3)

# Split data into train and test sets
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    images, categories_encoded, test_size=0.2, random_state=42
)
_, _, y_train_fresh, y_test_fresh = train_test_split(
    images, freshness, test_size=0.2, random_state=42
)
_, _, y_train_life, y_test_life = train_test_split(
    images, lifespan, test_size=0.2, random_state=42
)

# Save preprocessed data
np.savez("preprocessed_data.npz",
         X_train=X_train, X_test=X_test,
         y_train_cat=y_train_cat, y_test_cat=y_test_cat,
         y_train_fresh=y_train_fresh, y_test_fresh=y_test_fresh,
         y_train_life=y_train_life, y_test_life=y_test_life)

print("Data preprocessing completed. Saved to 'preprocessed_data.npz'.")
