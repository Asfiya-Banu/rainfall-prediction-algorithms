# Rainfall Prediction Project

This project utilizes Databricks to analyze historical rainfall data retrieved from Azure Blob Storage. The goal is to predict annual rainfall patterns using machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Data Retrieval](#data-retrieval)
- [Data Exploration and Cleaning](#data-exploration-and-cleaning)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

This project demonstrates how to use Databricks for data analysis and machine learning tasks. It retrieves historical rainfall data from Azure Blob Storage, performs data exploration, applies preprocessing techniques, selects relevant features, trains machine learning models, evaluates model performance, and visualizes results.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Install dependencies:**
   - Ensure Python 3.x and pip are installed.
   - Install required libraries listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

## Data Retrieval

The historical rainfall data (`Rainfall.csv`) is stored in Azure Blob Storage. The script connects to Azure Blob Storage using the Azure Storage Blob library, retrieves the CSV file, and loads it into a Pandas DataFrame for further analysis.

```python
# Install Azure Storage Blob library
%pip install azure-storage-blob

from azure.storage.blob import BlobServiceClient

# Azure Blob Storage credentials and container info
account_url = "https://rainfallpridictor.blob.core.windows.net"
sas_token = "sp=racwdli&st=2024-06-14T10:42:43Z&se=2024-08-30T18:42:43Z&spr=https&sv=2022-11-02&sr=c&sig=leVZ3NKeNInfmUPcRraX8NZpaL9LdwjBEuYKBpCJh0A%3D"
container_name = "rainfalldata"

# Connect to Blob service
blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
container_client = blob_service_client.get_container_client(container_name)

# Download CSV file
blob_name = "Rainfall.csv"
blob_client = container_client.get_blob_client(blob_name)
blob_data = blob_client.download_blob().content_as_text()

import pandas as pd

# Load CSV data into Pandas DataFrame
df = pd.read_csv(blob_data)
print(df.head())
```

## Data Exploration and Cleaning

- Perform exploratory data analysis (EDA) to understand the structure and characteristics of the dataset.
- Clean the data by handling missing values and ensuring data consistency.

```python
# Data exploration and cleaning
print(df.shape)
print(df.info())
print(df.describe().T)
print(df.isnull().sum())

# Clean data: fill missing values with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)

print(df.isnull().sum().sum())
```

## Feature Selection

Identify and select relevant features for training machine learning models. This involves choosing numeric features and possibly encoding categorical variables.

```python
# Feature selection
features = list(df.select_dtypes(include=np.number).columns)
features.remove('YEAR')  # Assuming 'YEAR' is not used as a feature
print(features)
```

## Model Training and Evaluation

Train machine learning models (e.g., Logistic Regression, XGBoost, SVM) to predict annual rainfall patterns. Evaluate model performance using appropriate metrics such as ROC AUC score and confusion matrix.

```python
# Model training and evaluation (example with Logistic Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# Split data into training and validation sets
X = df[features]
y = df['ANNUAL']  # Assuming 'ANNUAL' is the target variable

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on validation set and evaluate
y_pred = model.predict(X_val)
roc_auc = roc_auc_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print(f'ROC AUC Score: {roc_auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
```

## Visualization

Visualize the data distributions, correlations, and model evaluation results using Matplotlib and Seaborn.

```python
# Visualization example: Histogram of 'ANNUAL' values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['ANNUAL'], bins=20, kde=True)
plt.title('Distribution of Annual Rainfall')
plt.xlabel('Rainfall')
plt.ylabel('Frequency')
plt.show()
```

## Usage

1. Open `analysis.ipynb` in your Databricks workspace.
2. Run each cell sequentially to execute the analysis and model training.
3. Review the results and visualizations generated in the notebook.

## Dependencies

- `azure-storage-blob`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
