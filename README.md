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

To ensure universal deployment across various Linux distributions (Ubuntu, CentOS, Fedora, etc.), the project is containerized using Docker.

### Universal Deployment with Docker

### Prerequisites
*   [Docker](https://www.docker.com/get-started)

### Building the Docker Image
```bash
docker build -t phishing-detector .
```

### Running the Project (Inside Docker)

The Docker image provides a consistent environment. You will still need to provide the raw dataset file (`dataset_full.csv`) inside the container to run the scripts.

1.  **Download the Dataset**: The raw dataset (`dataset_full.csv`) and the cleaned, scaled dataset (`phishing_dataset_clean.csv`) are too large for direct inclusion in this repository (exceeding GitHub's 100MB limit). You must download the raw data and place it in the project directory before building the image or mount it as a volume.

**Raw Dataset Source**: [https://github.com/GregaVrbancic/Phishing-Dataset](https://github.com/GregaVrbancic/Phishing-Dataset)

2.  **Prepare Data**: Run the data preparation script inside the container.
    ```bash
    docker run --rm -v $(pwd):/app phishing-detector python data_preparation.py
    ```
3.  **Train and Evaluate Model**: Run the training script inside the container. This will train the model, evaluate it, and save the results.
    ```bash
    docker run --rm -v $(pwd):/app phishing-detector python model_training.py
    ```

The trained model (`phishing_detection_model_trained.keras`) and evaluation results will be saved to your local project directory.

### Local Setup (Alternative)

If you prefer a local setup, follow these steps:

1.  **Prerequisites**: Python 3.x and `pip`.
2.  **Installation**: Install dependencies from `requirements.txt`.
    ```bash
    pip3 install -r requirements.txt
    ```
3.  **Running the Project**: Follow steps 1-3 from the "Running the Project (Inside Docker)" section, replacing the `docker run` commands with direct `python3` calls.
    ```bash
    python3 data_preparation.py
    python3 model_training.py
    ```
1.  **Download the Dataset**: The raw dataset (`dataset_full.csv`) and the cleaned, scaled dataset (`phishing_dataset_clean.csv`) are too large for direct inclusion in this repository (exceeding GitHub's 100MB limit). You must download the raw data and run the `data_preparation.py` script to generate the cleaned data and the necessary NumPy arrays for training.

**Raw Dataset Source**: [https://github.com/GregaVrbancic/Phishing-Dataset](https://github.com/GregaVrbancic/Phishing-Dataset)
2.  **Prepare Data**: Run the data preparation script.
    ```bash
    python3 data_preparation.py
    ```
3.  **Train and Evaluate Model**: Run the training script. This will train the model, evaluate it, and save the results.
    ```bash
    python3 model_training.py
    ```

The trained model (`phishing_detection_model_trained.keras`) and evaluation results will be saved to the project directory.
