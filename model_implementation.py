import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Reshape

# Load the processed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Reshape data for 1D CNN: (samples, timesteps, features)
# Our features are already a sequence of 111 items, so we treat them as 111 timesteps with 1 feature each.
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the 1D CNN Model
model = Sequential([
    # Input layer: 111 features, 1 channel
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification output
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Save the model summary to a file for documentation
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("1D CNN Model defined and compiled successfully.")
print(f"Input shape for CNN: {X_train_reshaped.shape[1]}")
print("Model summary saved to model_summary.txt")

# Save the compiled model structure (not trained weights) for the next phase
model.save('phishing_detection_model_initial.keras')
