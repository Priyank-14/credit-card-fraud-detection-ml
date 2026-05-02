# ===============================================================
#      CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# ===============================================================

# This Project aims to detect fraudulent credit card transactions
# using supervised machine learning techniques.

"""
Workflow followed:
1. Load dataset
2. Data understanding and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature scaling
5. Handling imbalanced data using SMOTE
6. Model training (Logistic Regression & Random Forest)
7. Model evaluation and comparison
"""

# =========================================
#           REQUIRED LIBRARIES
# =========================================
# - Data handling (pandas, numpy)
# - Visualization (matplotlib, seaborn)
# - Machine learning (sklearn)
# - Handling imbalanced data (SMOTE)
# - File path management (os)
# =========================================

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


# ========================================
#               PATH SETUP
# ========================================

"""
This section ensures that:
- The dataset is correctly loaded from the 'data' folder.
- The output graphs are saved in the 'output' folder.
- Code works regardless of system location (portable code).
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

data_path = os.path.join(project_root, "data", "creditcard.csv")
output_path = os.path.join(project_root, "output")

os.makedirs(output_path, exist_ok=True)

# =============================================
#         STEP 1: LOADING THE DATASET
# =============================================

"""
The dataset contains anonymized transaction features.
'Class' column is the target variable:
    0 -> Normal Transaction
    1 -> Fraudulent Transaction
"""

print("Loading dataset from:", data_path)

data = pd.read_csv(data_path)

# ============================================================
# DATASET SIZE OPTIMIZATION
# ============================================================
"""
The original dataset is large (~284k rows). After applying SMOTE,
the dataset size increases significantly, leading to high memory
usage and slower training.

This preserves data distribution while improving performance.
"""
data = data.sample(n=40000, random_state=42)

print("\nUsing optimized dataset size:", data.shape)

# Basic dataset information
print("\nFirst Five Rows of Dataset:")
print(data.head())

print("\nDataset shape (rows, columns):", data.shape)

# =================================================
#   STEP 2: DATA UNDERSTANDING AND PREPROCESSING 
# =================================================

print("\nDataset Info:")
print(data.info())

print("\nChecking Missing Values:")
print(data.isnull().sum())

print("\nChecking Duplicate rows:")
duplicates = data.duplicated().sum()
print("Number of duplicates:", duplicates)

if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates removed")
else:
    print("No duplicate records found")

print("\nClass distribution (0=Normal, 1=Fraud):")
print(data['Class'].value_counts())

# ==========================================================
#          STEP 3: EXPLORATORY DATA ANALYSIS(EDA)
# ==========================================================

print("\nGenerating Visualizations!")

plt.figure()
sns.countplot(x="Class", data=data)
plt.title("Class Distribution")
plt.savefig(os.path.join(output_path, "class_distribution.png"))
plt.show()

plt.figure()
sns.histplot(data['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.savefig(os.path.join(output_path, "amount_distribution.png"))
plt.show()

plt.figure()
sns.boxplot(x=data['Amount'])
plt.title("Outlier Detection (Amount)")
plt.savefig(os.path.join(output_path, "amount_outliers.png"))
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))
plt.show()

# =====================================================
#           STEP 4: DATA PREPARATION
# =====================================================

X = data.drop("Class", axis=1)
y = data["Class"]

# ========================================
#       STEP 5: FEATURE SCALING
# ========================================

print("\nApplying Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================
#       STEP 6: HANDLING IMBALANCED DATA
# ==============================================

"""
SMOTE is applied to handle class imbalance.

Instead of fully balancing the dataset (which increases memory usage),
partial resampling is used.

sampling_strategy=0.5 means:
Fraud class will be increased to 50% of normal class.

This balances:
- Model performance
- Computational efficiency
"""

print("\nBefore SMOTE:")
print(y.value_counts())

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nAfter SMOTE:")
print(pd.Series(y_resampled).value_counts())

plt.figure()
sns.countplot(x=y_resampled)
plt.title("Balanced dataset after SMOTE")
plt.savefig(os.path.join(output_path, "balanced_data.png"))
plt.show()

# ==============================================
#           STEP 7: TRAIN-TEST SPLIT
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ========================================
#          STEP 8: MODEL TRAINING
# ========================================

print("\nTraining Models.....")

# Logistic Regression
"""
Baseline classification model.

max_iter=200:
Ensures proper convergence
"""
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Random Forest
"""
Ensemble model for improved accuracy.

Optimized parameters:
- n_estimators=15 → fewer trees for speed
- max_depth=8 → limits complexity
- n_jobs=-1 → parallel processing
"""
rf_model = RandomForestClassifier(
    n_estimators=15,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

# ================================
#       STEP 9: PREDICTIONS
# ================================

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ======================================
#       STEP 10: MODEL EVALUATION
# ======================================

print("\n===== Logistic Regression Results ====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test, lr_pred))
print("Classification Report:\n",classification_report(y_test, lr_pred))

print("\n===== Random Forest Results ====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test, rf_pred))
print("Classification Report:\n",classification_report(y_test, rf_pred))

# ================================================
#           STEP 11: MODEL COMPARISON
# ================================================

models = ['Logistic Regression', 'Random Forest']
accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, rf_pred)
]

plt.figure()
plt.bar(models, accuracies) 
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(output_path, "model_comparison.png"))
plt.show()

# ====================================
#           END OF PROJECT
# ====================================

print("\nProject executed successfully!")
