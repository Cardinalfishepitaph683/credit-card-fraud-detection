# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-End Machine Learning Pipeline for Detecting Fraudulent Credit Card Transactions using SMOTE, Random Forest, and XGBoost.

This project builds a complete machine learning workflow to identify fraudulent credit card transactions from an extremely imbalanced dataset.  
The pipeline includes data preprocessing, class imbalance handling using SMOTE, training multiple models, evaluating performance, and saving the best model.

---

## Problem Statement

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared to normal transactions.

In the dataset used for this project, fraud cases represent **less than 1% of all transactions**, making it a **severely imbalanced classification problem**.

Traditional machine learning models often fail in such scenarios, so techniques like **SMOTE (Synthetic Minority Oversampling Technique)** are used to balance the training data and improve fraud detection performance.

---

## Objective

The main objective of this project is to:

- Build a robust machine learning pipeline for fraud detection
- Handle extreme class imbalance effectively
- Compare multiple machine learning models
- Evaluate model performance using classification metrics
- Save the best performing model for future predictions

---

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset Source  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Characteristics:

- Total Transactions: **284,807**
- Fraud Cases: **492**
- Features: **31 columns**
- Highly imbalanced dataset

Target Variable:

`Class`
0 → Normal Transaction
1 → Fraud Transaction


Note:  
The dataset is **not included in this repository due to GitHub file size limits**.  
Download it from Kaggle and place the CSV file inside the `data/` folder.

---

## Technologies Used

### Programming
- Python

### Data Science Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

### Tools
- Git
- GitHub
- Joblib

---

## Machine Learning Models

This project trains and compares multiple machine learning models:

- Logistic Regression (Baseline Model)
- Random Forest (Ensemble Model)
- XGBoost (Gradient Boosting Model)

The models are evaluated using classification metrics to determine the best performing model.

---

## Project Workflow

The project follows a structured machine learning pipeline:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Class Distribution Analysis
4. Train-Test Split
5. Handling Class Imbalance using SMOTE
6. Model Training
7. Model Comparison
8. Model Evaluation
9. Best Model Selection
10. Model Saving

---

## Evaluation Metrics

Because fraud detection involves imbalanced data, accuracy alone is not sufficient.

The models are evaluated using:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Special focus is given to **Recall and F1-score for the fraud class**, since detecting fraudulent transactions is the primary goal.

---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                    # Dataset folder (place creditcard.csv here)
│
├── models/                  # Saved trained models
│
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── notebooks/               # Future experimentation notebooks
│
├── requirements.txt         # Project dependencies
│
└── README.md                # Project documentation
```

---

## Results

The machine learning pipeline successfully detects fraudulent credit card transactions by handling class imbalance using SMOTE and training multiple classification models.

Ensemble models such as **Random Forest and XGBoost** improve the model's ability to detect fraudulent transactions compared to a simple baseline model.

The final model achieves strong fraud detection performance with improved **Recall and F1-score for the minority fraud class**.

---

## How to Run the Project

### 1. Install dependencies
pip install -r requirements.tx

### 2. Run data preprocessing
python src/data_preprocessing.py

### 3. Train the machine learning models
python src/train_model.py

### 4. Evaluate model performance
python src/evaluate_model.py

---

## Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature scaling and feature selection
- Model performance visualization
- Experiment tracking using MLflow
- Deploying the model using FastAPI
- Containerizing the application using Docker

---

## Author

Bhushan Patil  
AI / Machine Learning Engineer  
Pune, Maharashtra, India
