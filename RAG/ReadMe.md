# **Project Workflow**

Credit card fraud is a major financial threat worldwide. Millions of transactions occur daily, and detecting fraud manually is nearly impossible.

This project builds a Machine Learning pipeline that:

* Processes transaction data

* Handles class imbalance

* Trains multiple ML models

* Evaluates performance using fraud detection metrics

The objective is to maximize fraud detection while minimizing false positives.

* **Problem Statement**


The dataset contains 284,807 transactions, but only 492 transactions are fraudulent.

This means:

* 99.8% transactions are legitimate

* 0.2% transactions are fraud

Such datasets create a class imbalance problem, which can cause models to predict everything as non-fraud.

This project addresses this issue using resampling techniques and proper evaluation metrics.

# **Project Architecture**

![][image1]

---

## 

## **1Data Understanding**

* ## Load dataset 

* ## Inspect data structure 

* ## Identify class imbalance 

## ---

## **2 Exploratory Data Analysis (EDA)**

## EDA includes:

* ## Fraud vs Non-Fraud distribution 

* ## Transaction amount analysis 

* ## Feature correlation analysis 

* ## Outlier detection 

## ---

## **3Data Preprocessing**

## Steps performed:

* ## Feature scaling 

* ## Train-test split 

* ## Handling missing values 

* ## Data balancing using **SMOTE** 

## ---

# **4.Handling Imbalanced Data**

## Fraud detection datasets are **highly imbalanced**.

## To address this issue:

## **SMOTE (Synthetic Minority Oversampling Technique)** was used.

## SMOTE generates **synthetic fraud samples** to help the model learn fraud patterns better.

## ---

# **5.Machine Learning Models Used**

## The following models were trained and evaluated:

### **1Logistic Regression**

## Baseline classification model used for binary prediction.

### **2 Decision Tree**

## Captures non-linear feature interactions.

### **3XGBoost Classifier**

## Gradient boosting algorithm known for high performance on structured datasets.

## 

## 

---

# Model Evaluation Metrics

Because the dataset is imbalanced, accuracy alone is not reliable.

The following metrics were used:

* Accuracy

* Precision

* Recall

* F1 Score

* Confusion Matrix

* ROC-AUC Score

Special focus was given to Recall, since detecting fraud is more important than missing it.

---

# **Results**

The models were evaluated and compared based on their ability to correctly identify fraudulent transactions.

Key observations:

* Handling imbalance significantly improved model performance.

* Ensemble methods like XGBoost performed better than basic models.

* Recall improved after applying SMOTE.

---

# Technologies Used

* Python

* Jupyter Notebook

* Pandas

* NumPy

* Matplotlib

* Seaborn

* Scikit-learn

* Imbalanced-learn

* XGBoost

---

# Project Structure

Credit-Card-Fraud-Detection  
│  
├── Credit\_card\_fraud\_detection.ipynb  
├── README.md  
└── dataset  
---

---

[image1]: <data:image/png;base64,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>