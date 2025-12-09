import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = '/home/ubuntu/dataset_full.csv'
df = pd.read_csv(file_path)

# The last column 'phishing' is the target variable (y)
# All other columns are features (X)
X = df.drop('phishing', axis=1)
y = df['phishing']

# Handle missing values: The dataset uses -1 to represent missing or non-applicable values.
# We will replace -1 with NaN and then use SimpleImputer to fill NaNs with the mean of the column.
X = X.replace(-1, np.nan)

# Impute missing values (NaNs) with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Scale the features to a range of [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed_df)

# Split the data into training and testing sets
# We will use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save the processed data for the next phase
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"Data loaded and processed successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"The number of features is: {X_train.shape[1]}")
