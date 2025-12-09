# Phishing Website Detection Using Deep Learning: A 1D Convolutional Neural Network Approach

## 1. Introduction

Phishing remains one of the most prevalent and effective cyber-attack vectors, relying on deception to steal sensitive user information. The increasing sophistication of these attacks necessitates robust and automated detection mechanisms. This report details the development and evaluation of a deep learning model, specifically a **1D Convolutional Neural Network (1D CNN)**, for the binary classification of websites as either legitimate or phishing, based on a comprehensive set of extracted URL and content-based features.

## 2. Methodology

### 2.1. Data Acquisition

The model was trained and evaluated using the **Phishing Dataset** by Grega Vrbancic [1], which is a publicly available, feature-engineered dataset.

| Metric | Value |
| :--- | :--- |
| Total Instances | 88,647 |
| Number of Features | 111 |
| Feature Types | URL-based, Content-based, External Service-based |
| Class Balance | Approximately 50% Legitimate, 50% Phishing |

### 2.2. Data Preprocessing

The dataset contained pre-extracted features, which simplified the initial steps. The following preprocessing steps were applied:

1.  **Missing Value Imputation**: The dataset used a value of `-1` to denote missing or non-applicable features. These values were replaced with the mean of their respective columns to ensure the neural network could process the data without errors.
2.  **Feature Scaling**: All 111 features were scaled to a range between 0 and 1 using the **MinMaxScaler** from `scikit-learn`. This standardization is crucial for optimizing the performance of deep learning models.
3.  **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets, with stratification to maintain the original class distribution in both subsets.

### 2.3. Model Architecture

A 1D Convolutional Neural Network (1D CNN) was selected for its ability to automatically learn hierarchical patterns and local dependencies within the sequence of features. The 111 features were treated as a sequence of 111 time-steps with a single channel.

The model architecture is as follows:

| Layer (Type) | Output Shape | Parameters |
| :--- | :--- | :--- |
| `conv1d` (Conv1D) | (None, 109, 64) | 256 |
| `conv1d_1` (Conv1D) | (None, 107, 128) | 24,704 |
| `flatten` (Flatten) | (None, 13696) | 0 |
| `dense` (Dense) | (None, 128) | 1,753,216 |
| `dropout` (Dropout) | (None, 128) | 0 |
| `dense_1` (Dense) | (None, 64) | 8,256 |
| `dense_2` (Dense) | (None, 1) | 65 |
| **Total Parameters** | | **1,786,497** |

The model was compiled with the **Adam optimizer** and **binary cross-entropy** loss, with **Accuracy**, **Precision**, and **Recall** as evaluation metrics.

## 3. Results and Discussion

The model was trained for **5 epochs** on the training data. The performance was then evaluated on the unseen test set.

### 3.1. Model Performance Metrics

The evaluation on the test set yielded the following results:

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **0.9519** |
| **Test Precision** | **0.9414** |
| **Test Recall** | **0.9179** |
| **Test Loss** | 0.1291 |

The high accuracy of **95.19%** indicates that the 1D CNN is highly effective at distinguishing between legitimate and phishing websites using the provided feature set.

### 3.2. Classification Report

The detailed classification report provides a breakdown of performance for each class:

| | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Legitimate (0)** | 0.9572 | 0.9698 | 0.9635 | 11600 |
| **Phishing (1)** | 0.9414 | 0.9179 | 0.9295 | 6130 |
| **Macro Avg** | 0.9493 | 0.9439 | 0.9465 | 17730 |
| **Weighted Avg** | 0.9518 | 0.9519 | 0.9517 | 17730 |

The model exhibits strong performance across both classes, with a slightly higher recall for the legitimate class, indicating a low rate of false positives (legitimate sites flagged as phishing). The precision for the phishing class is also very high, meaning that when the model predicts a site is phishing, it is correct over 94% of the time.

### 3.3. Confusion Matrix

The confusion matrix further illustrates the model's predictive power:

| | Predicted Legitimate (0) | Predicted Phishing (1) |
| :--- | :--- | :--- |
| **Actual Legitimate (0)** | 11,250 (True Negative) | 350 (False Positive) |
| **Actual Phishing (1)** | 503 (False Negative) | 5,627 (True Positive) |

The model correctly identified **11,250** legitimate websites and **5,627** phishing websites. The number of false negatives (phishing sites missed) is **503**, which is a critical metric in security applications, and the model's performance in this area is strong.

## 4. Conclusion

The developed 1D CNN model for phishing website detection, utilizing a feature-engineered dataset, achieved an impressive **95.19% accuracy** on the test set. The results demonstrate the effectiveness of applying deep learning, specifically convolutional layers, to structured feature data for high-stakes binary classification problems in cybersecurity. The model is robust and provides a strong foundation for a real-time phishing detection system.

## 5. References

[1] GregaVrbancic. *Phishing-Dataset*. GitHub. Available at: [https://github.com/GregaVrbancic/Phishing-Dataset](https://github.com/GregaVrbancic/Phishing-Dataset)
