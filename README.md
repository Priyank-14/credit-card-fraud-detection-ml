![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/ML-Project-green)
![Status](https://img.shields.io/badge/Status-Completed-success)


# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 About the Project
This project focuses on detecting fraudulent credit card transactions using machine learning techniques.

The dataset is highly imbalanced, making fraud detection a challenging task. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied to improve the model’s ability to identify minority class (fraud) transactions.

The project implements a complete machine learning pipeline including data preprocessing, exploratory data analysis, feature scaling, model training, and evaluation.

---

## 🚀 Project Highlights

- Handles highly imbalanced dataset using SMOTE  
- Achieves ~99% accuracy with Random Forest  
- Focuses on recall to minimize fraud detection errors  
- End-to-end ML pipeline (EDA → preprocessing → modeling → evaluation)  

---

## 🧠 Machine Learning Approach

- Type: Supervised Learning (Binary Classification)

### Algorithms Used:
- Logistic Regression — Baseline linear model  
- Random Forest — Ensemble model for improved performance  

### Additional Technique:
- SMOTE — Handles class imbalance by generating synthetic samples  

---

## 📊 Dataset

- Name: Credit Card Fraud Detection Dataset  
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

### Key Characteristics:
- Highly imbalanced dataset  
- PCA-transformed features (V1–V28) for privacy  
- Target variable:
  - 0 → Normal Transaction
  - 1 → Fraudulent Transaction  

---

<h2>📁 Project Structure</h2>

<pre>
credit-card-fraud-detection-ml/
│
├── src/
│   └── fraud_detection.py
│
├── output/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── amount_outliers.png
│   ├── correlation_heatmap.png
│   ├── balanced_data.png
│   └── model_comparison.png
│
├── report/
│   ├── Project_Report.docx
│   └── presentation.pptx
│
├── requirements.txt
└── README.md
</pre>

---

## ⚙️ How to Run

1. Install dependencies  
pip install -r requirements.txt  

2. Run the project  
python src/fraud_detection.py  

---

## 📈 Output

The project generates the following visualizations:

- Class distribution (before balancing)  
- Transaction amount distribution  
- Outlier detection (boxplot)  
- Correlation heatmap  
- Balanced dataset visualization (after SMOTE)  
- Model performance comparison  

All outputs are saved in the `output/` directory.


## 📸 Output Preview

### Class Distribution
![Class Distribution](output/class_distribution.png)

### SMOTE Balanced Data
![Balanced Data](output/balanced_data.png)

### Model Comparison
![Model Comparison](output/model_comparison.png)
---

## 📊 Results

Model | Accuracy | Recall  
------|----------|--------  
Logistic Regression | ~96% | ~0.93  
Random Forest | ~99% | ~0.98  

### 🔍 Key Insight
In fraud detection systems, **Recall is more important than Accuracy** because missing fraudulent transactions can lead to significant financial loss.

---

## 📌 Key Learnings

- Handling highly imbalanced datasets using SMOTE  
- Importance of recall in fraud detection problems  
- Feature scaling and preprocessing techniques  
- Model comparison using confusion matrix and classification metrics  

---

## 🔮 Future Improvements

- Real-time fraud detection system  
- Deployment as a web application  
- Use of deep learning and anomaly detection models  

---

## 🧾 Requirements

pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
imbalanced-learn  

---

## 👨‍💻 Author

Priyank Sinha  
B.Tech CSE (UPES)  

GitHub: https://github.com/Priyank-14  

---

## ⭐ Note

This project is developed for academic and learning purposes and demonstrates practical implementation of machine learning techniques in fraud detection.
