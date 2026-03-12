# Credit Card Fraud Detection

Machine Learning | Fraud Detection | Imbalanced Dataset | SMOTE

End-to-End Machine Learning Pipeline for Detecting Fraudulent Credit Card Transactions.

This project builds a machine learning pipeline to identify fraudulent credit card transactions using an imbalanced dataset and SMOTE-based oversampling.

---

## Objective

To detect fraudulent credit card transactions and handle class imbalance effectively using machine learning.

---

## Dataset

Note: The dataset is not included in this repository due to GitHub file size limits.
Download it from Kaggle and place it inside the `data/` folder.

Dataset Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Credit Card Fraud Detection Dataset

Records: 284,807 transactions  
Features: 31 columns  

Target Variable:

`Class`  
- `0` → Normal transaction  
- `1` → Fraud transaction  

Fraud transactions are extremely rare, making this an imbalanced classification problem.

---

## Tech Stack

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
imbalanced-learn  
Joblib  
Git & GitHub  

---

## Project Workflow

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Class Distribution Analysis  
4. Train-Test Split  
5. SMOTE Oversampling  
6. Model Training  
7. Model Evaluation  
8. Model Saving  

---

## Model Used

Logistic Regression

---

## Key Learning

This project focuses on imbalanced classification, where fraud transactions make up less than 1% of the dataset.  
SMOTE is used to generate synthetic fraud samples and improve fraud detection performance.

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/                  # Dataset
├── models/                # Saved model
├── src/                   # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── notebooks/             # Future notebooks
├── requirements.txt
└── README.md

## Result

The model detects fraudulent transactions using logistic regression and handles severe class imbalance using SMOTE.

The trained model achieves strong fraud detection performance with improved recall and F1-score for the minority fraud class.

## How to Run

Install dependencies

pip install -r requirements.txt

Run preprocessing

python src/data_preprocessing.py

Train the model

python src/train_model.py

Evaluate the model

python src/evaluate_model.py

## Author

Bhushan Patil  
AI / Machine Learning Engineer
