# Telco Customer Churn Prediction

An end-to-end **data analytics and machine learning project** focused on understanding customer churn and predicting customers who are at risk of leaving a telecommunications service.

The project combines exploratory data analysis, data preprocessing, visualization, machine learning, model evaluation, and model explainability to generate both predictive results and actionable business insights.

---

## 📌 Project Overview

Customer churn is an important business problem for telecommunications companies because losing customers can negatively affect revenue and customer lifetime value.

This project analyzes customer demographics, tenure, services, contract characteristics, payment methods, and billing behavior to identify patterns associated with customer churn.

The overall workflow is:

**Business Understanding → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Machine Learning → Model Evaluation → Explainability → Business Insights**

---

## 🎯 Objectives

* Understand the characteristics of customers who churn.
* Identify customer segments with higher churn rates.
* Explore factors associated with customer retention and churn.
* Clean and prepare customer data for analysis.
* Perform exploratory data analysis and visualization.
* Build a leakage-safe machine learning pipeline.
* Train and compare classification models.
* Evaluate models using business-relevant metrics.
* Identify important features associated with churn.
* Translate analytical findings into actionable retention strategies.
* Save the trained model and analytical outputs for future reuse.

---

## 📊 Dataset

The project uses the **Telco Customer Churn dataset** obtained from Kaggle.

The dataset contains **7,043 customer records** and includes information related to:

* Customer demographics
* Customer tenure
* Contract type
* Internet services
* Additional services
* Payment methods
* Monthly charges
* Total charges
* Churn status

### Target Variable

`Churn`

* `Yes` — Customer churned
* `No` — Customer did not churn

The original dataset is stored separately under `data/raw/`.

---

## 🔐 Data Leakage Prevention

A key part of this project was preventing information leakage from the target-related fields.

The following columns were excluded before model training:

* `Churn Value`
* `Churn Score`
* `Churn Reason`

These variables contain information directly related to the churn outcome and therefore should not be used as predictive features.

Geographic fields were also removed to keep the model focused on customer behavior, services, contracts, and account characteristics:

* `City`
* `Zip Code`
* `Lat Long`
* `Latitude`
* `Longitude`

This results in a more meaningful and business-oriented feature set.

---

## 🔍 Business Questions

The analysis investigates questions such as:

1. What proportion of customers churn?
2. Which contract types have the highest churn?
3. How does customer tenure relate to churn?
4. Are customers with higher monthly charges more likely to churn?
5. Which payment methods are associated with higher churn?
6. Which internet services are associated with higher churn?
7. Which customer segments should receive retention attention?
8. Which features are most useful for predicting churn?
9. Which machine learning model provides the best predictive performance?

---

## 📈 Exploratory Data Analysis

The project includes visual analysis of:

* Overall churn distribution
* Churn by contract type
* Tenure distribution by churn status
* Churn by monthly-charge bands
* Churn by payment method
* Churn by internet service

### Key Observations

The analysis indicates higher churn concentration among:

* **Month-to-month customers**
* **Newer customers**
* Customers with **higher monthly charges**
* Customers using **electronic check** as their payment method
* Customers using **fiber-optic internet service**

These patterns provide useful areas for targeted customer-retention strategies.

---

## 🤖 Machine Learning

Two classification models were trained and evaluated:

* Logistic Regression
* Random Forest

### Preprocessing Pipeline

The machine learning pipeline includes:

* Median imputation for numerical variables
* Standard scaling for numerical variables
* Most-frequent imputation for categorical variables
* One-hot encoding for categorical variables
* Class-weighted model training to address class imbalance

A stratified train/test split was used to maintain the churn-class distribution across the training and testing datasets.

---

## 📊 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Because the business objective is to identify customers who may leave, **Recall for the churn class and ROC-AUC** are particularly important.

### Model Performance

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Logistic Regression** | **0.743** | **0.510** | **0.783** | **0.618** | **0.849** |
| Random Forest           |     0.779 |     0.571 |     0.663 |     0.614 |     0.838 |

### Best Model

**Logistic Regression** achieved the highest ROC-AUC:

**ROC-AUC = 0.849**

Although Random Forest achieved higher accuracy, Logistic Regression provides stronger ROC-AUC and recall for identifying churned customers.

For a retention-focused use case, this makes Logistic Regression the preferred model in this project.

---

## 🧠 Model Explainability

Feature importance for the selected Logistic Regression model is analyzed using model coefficients.

The strongest predictors include:

* Customer tenure
* Dependents
* Contract type
* Internet service
* Monthly charges
* Total charges
* Paperless billing

The analysis provides an interpretable view of which customer characteristics are associated with increased or decreased churn risk.

> Feature importance represents model associations and should not automatically be interpreted as proof of causation.

---

## 💡 Business Insights

The analysis suggests that retention efforts should particularly focus on:

### 1. New Customers

Customers with shorter tenure show greater churn risk.

**Potential action:**
Introduce stronger onboarding, early engagement, and proactive support during the first months of service.

### 2. Month-to-Month Customers

Month-to-month customers represent an important high-risk segment.

**Potential action:**
Offer incentives or benefits that encourage customers to move toward longer-term contracts.

### 3. Higher Monthly Charges

Customers in higher monthly-charge segments show elevated churn.

**Potential action:**
Review pricing, perceived value, and service quality for higher-paying customers.

### 4. Electronic Check Users

Electronic-check customers show higher churn levels.

**Potential action:**
Encourage convenient automatic payment methods through incentives or simplified enrollment.

### 5. Fiber-Optic Customers

Fiber-optic customers show elevated churn in the analyzed dataset.

**Potential action:**
Investigate service quality, customer expectations, technical issues, and perceived value within this segment.

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── Telco_customer_churn.xlsx
│
├── notebooks/
│   └── 01_telco_churn_analysis.ipynb
│
├── outputs/
│   ├── best_churn_model.joblib
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   │
│   └── figures/
│       ├── eda.png
│       ├── tenure_by_churn.png
│       ├── business_churn_drivers.png
│       ├── roc_curves.png
│       ├── logistic_regression_confusion_matrix.png
│       └── random_forest_confusion_matrix.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Project Outputs

The `outputs/` directory contains reusable analytical and modeling results:

* `best_churn_model.joblib` — trained best-performing model pipeline
* `model_comparison.csv` — model evaluation results
* `feature_importance.csv` — ranked model features
* `figures/` — EDA, ROC curve, and confusion matrix visualizations

---

## 🛠️ Technologies

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development

* Jupyter Notebook
* Git
* GitHub

---

## 🚀 Future Improvements

Potential future improvements include:

* Probability threshold optimization
* Cost-sensitive decision analysis
* Customer risk segmentation
* Hyperparameter tuning
* Model calibration
* Additional classification algorithms
* Model monitoring
* Data drift detection
* Interactive Streamlit application
* Production API deployment
* Integration with customer retention workflows

---

## 👩‍💻 Author

**Javaria Qadeer**

Data Science | Machine Learning | Data Analytics

This project is part of my professional portfolio development toward **Data Analyst and Data Scientist roles**.
