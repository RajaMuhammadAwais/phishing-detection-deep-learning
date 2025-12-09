# Phishing Website Detection Using Deep Learning (1D CNN)

This project implements a deep learning model, specifically a **1D Convolutional Neural Network (1D CNN)**, to detect phishing websites based on a comprehensive set of extracted URL and content-based features.

The model achieved an **accuracy of 95.19%** on the test set, demonstrating its effectiveness in distinguishing between legitimate and malicious websites.

## Project Structure

| File/Directory | Description |
| :--- | :--- |
| `phishing_detection_report.md` | **Final Project Report** detailing the methodology, model architecture, training process, and full evaluation results. |
| `data_preparation.py` | Python script for loading the raw data, handling missing values, scaling features, and splitting the dataset into training and testing sets. |
| `model_implementation.py` | Python script for defining and compiling the 1D CNN model architecture. |
| `model_training.py` | Python script for training the model, evaluating its performance, and saving the trained model and evaluation metrics. |
| `model_summary.txt` | Text file containing the summary of the 1D CNN model architecture and parameter count. |
| `evaluation_metrics.txt` | Text file with the final model performance metrics (accuracy, precision, recall, classification report, and confusion matrix). |
| `.gitignore` | Specifies files to ignore, including the large dataset (`dataset_full.csv`), trained model weights (`*.keras`), and processed data arrays (`*.npy`). |

## Methodology Highlights

### Dataset
The model was trained on the **Phishing Dataset** by Grega Vrbancic, which contains 88,647 instances with 111 feature-engineered attributes derived from URL structure and web page content.

### Model
A 1D CNN was chosen to automatically learn patterns from the sequence of 111 features. The model architecture includes:
*   Two `Conv1D` layers (64 and 128 filters)
*   A `Flatten` layer
*   Two `Dense` layers (128 and 64 units) with a `Dropout` layer for regularization
*   A final `Dense` layer with a `sigmoid` activation for binary classification.

## Performance Results

The model was trained for 5 epochs. The key performance metrics on the test set are:

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **0.9519** |
| **Test Precision** | **0.9414** |
| **Test Recall** | **0.9179** |

For a detailed breakdown, please refer to the `phishing_detection_report.md` and `evaluation_metrics.txt` files.

## Setup and Reproduction

### Prerequisites
*   Python 3.x
*   `pip`

### Installation
```bash
pip3 install pandas scikit-learn tensorflow
```

### Running the Project
1.  **Download the Dataset**: Due to its size, the dataset (`dataset_full.csv`) is not included in the repository. You must download it separately from the source or use the provided `wget` command from the project history.
2.  **Prepare Data**: Run the data preparation script.
    ```bash
    python3 data_preparation.py
    ```
3.  **Train and Evaluate Model**: Run the training script. This will train the model, evaluate it, and save the results.
    ```bash
    python3 model_training.py
    ```

The trained model (`phishing_detection_model_trained.keras`) and evaluation results will be saved to the project directory.
