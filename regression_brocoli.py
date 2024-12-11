import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data for Broccoli
data = np.load("broccoli_preprocessed_data.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train_fresh = data["y_train_fresh"]
y_test_fresh = data["y_test_fresh"]
y_train_life = data["y_train_life"]
y_test_life = data["y_test_life"]

# Define the CNN-based regression model for Broccoli
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="linear")  # 2 outputs: freshness and lifespan
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="mse",
              metrics=["mae"])

# Save the best model during training
checkpoint = ModelCheckpoint("broccoli_best_regression_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    X_train, np.stack((y_train_fresh, y_train_life), axis=1),
    validation_data=(X_test, np.stack((y_test_fresh, y_test_life), axis=1)),
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint]
)

# Save the final model
model.save("broccoli_final_regression_model.keras")
print("Broccoli Regression Model training completed.")
