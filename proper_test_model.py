import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Function to load and preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)  # Read the image
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded.")
    img = cv2.resize(img, (128, 128))  # Resize to match input size of the model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to classify the image
def classify_image(model, img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    category = np.argmax(pred)  # Get the category with the highest probability
    categories = ["BROCOLI", "ONION", "PAPAYA"]
    return categories[category]


# Function to load the regression model
def load_regression_model(category):
    if category == "BROCOLI":
        return load_model("broccoli_final_regression_model.keras")
    elif category == "ONION":
        return load_model("onion_final_regression_model.keras")
    elif category == "PAPAYA":
        return load_model("papaya_final_regression_model.keras")
    else:
        raise ValueError("Unknown category. Choose from 'BROCOLI', 'ONION', or 'PAPAYA'.")


# Function to predict freshness and lifespan for the given image
def predict_freshness_and_lifespan(model, img_path, regression_model):
    img = preprocess_image(img_path)
    # Regression model predicts freshness and lifespan
    pred = regression_model.predict(img)
    freshness = pred[0][0]  # First output is freshness
    lifespan = pred[0][1]  # Second output is expected lifespan
    return freshness, lifespan


# Main function to execute the test
def main(img_path):
    # Load the classification model
    classification_model = load_model("final_classification_model.keras")

    # Step 1: Classify the image (get category)
    category = classify_image(classification_model, img_path)
    print(f"Predicted category: {category}")

    # Step 2: Load the corresponding regression model based on category
    regression_model = load_regression_model(category)

    # Step 3: Predict freshness and lifespan using the regression model
    freshness, lifespan = predict_freshness_and_lifespan(regression_model, img_path, regression_model)

    # Step 4: Print results
    print(f"Freshness: {freshness} days")
    print(f"Expected lifespan: {lifespan} days")


# If script is run directly
if __name__ == "__main__":
    # Define base path of the image
    img_path = input("Enter the full path of the image: ").strip()

    # Run the test
    main(img_path)
