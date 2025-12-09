import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the processed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Reshape data for 1D CNN: (samples, timesteps, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load the compiled model
model = load_model('phishing_detection_model_initial.keras')

# Train the model
print("Starting model training...")
history = model.fit(
    X_train_reshaped,
    y_train,
    epochs=5, # Reduced epochs to prevent timeout
    batch_size=64,
    validation_split=0.1, # Use 10% of training data for validation
    verbose=1
)
print("Model training complete.")

# Evaluate the model
print("Evaluating model performance...")
loss, accuracy, precision, recall = model.evaluate(X_test_reshaped, y_test, verbose=0)

# Make predictions
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Generate classification report and confusion matrix
report = classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)'], output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save evaluation metrics to a file
with open('evaluation_metrics.txt', 'w') as f:
    f.write("--- Model Evaluation ---\n")
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n\n")
    f.write("--- Classification Report ---\n")
    f.write(pd.DataFrame(report).transpose().to_markdown(numalign="left", stralign="left"))
    f.write("\n\n--- Confusion Matrix ---\n")
    f.write(np.array2string(conf_matrix, separator=', '))

# Save the trained model
model.save('phishing_detection_model_trained.keras')

print("Evaluation metrics saved to evaluation_metrics.txt")
print("Trained model saved to phishing_detection_model_trained.keras")
