
# Telco Customer Churn Prediction

An end-to-end **data analytics and machine learning project** focused on understanding customer churn and predicting customers who may be at risk of leaving a telecommunications service.

The project combines exploratory data analysis, data preprocessing, leakage-safe machine learning, model evaluation, model interpretation, and business-oriented insights.


## 📌 Project Overview

Customer churn is an important business problem for telecommunications companies because customer loss can negatively affect revenue and long-term customer relationships.

This project analyzes customer demographics, tenure, services, contract characteristics, payment methods, and billing behavior to identify patterns associated with customer churn.

The overall workflow is:

**Business Understanding → Data Cleaning → Exploratory Data Analysis → Preprocessing → Machine Learning → Model Evaluation → Model Interpretation → Business Insights**


## 🎯 Objectives

- Understand the characteristics of customers who churn.
- Identify customer segments with higher observed churn rates.
- Explore factors associated with customer retention and churn.
- Clean and prepare customer data for analysis.
- Perform exploratory data analysis and visualization.
- Build leakage-safe machine learning pipelines.
- Train and compare classification models.
- Evaluate models using multiple classification metrics.
- Select the best model using ROC-AUC.
- Interpret model coefficients and feature importance.
- Translate analytical findings into actionable retention strategies.
- Save the trained model and analytical outputs for future reuse.


## 📊 Dataset

The project uses the **Telco Customer Churn dataset** obtained from Kaggle.

The dataset contains **7,043 customer records** and includes information related to:

- Customer demographics
- Customer tenure
- Contract type
- Internet services
- Additional services
- Payment methods
- Monthly charges
- Total charges
- Churn status

The dataset is stored locally at:

```text
data/raw/Telco_customer_churn.xlsx
````

### Target Variable

`Churn`

* `Yes` — Customer churned
* `No` — Customer did not churn

For machine learning, the target is encoded as:

* `Yes = 1`
* `No = 0`

---

## 🧹 Data Cleaning and Preprocessing

The analysis includes:

* Standardization of column names
* Conversion of `TotalCharges` to numeric format
* Treatment of blank values as missing values
* Duplicate checking
* Removal of the customer identifier from predictive features
* Target encoding for `Churn`

Missing values are handled within the machine learning preprocessing pipeline rather than through preprocessing the entire dataset before the train/test split.

---

## 🔐 Data Leakage Prevention

Several fields contain information directly related to the churn outcome and were therefore excluded from model training:

* `Churn Value`
* `Churn Score`
* `Churn Reason`

`Customer ID` was also excluded because it is an identifier rather than a meaningful predictive feature.

Geographic fields were removed to keep the model focused on customer behavior, services, contracts, and account characteristics:

* `City`
* `Zip Code`
* `Lat Long`
* `Latitude`
* `Longitude`

### CLTV

`CLTV` was excluded from the predictive feature set to keep the analysis focused on customer characteristics and service/account behavior rather than directly incorporating a customer lifetime-value measure into the churn model.

The objective is to predict churn from customer and service characteristics rather than use an existing business-value metric as a predictor.

---

## 🔍 Business Questions

The analysis investigates questions such as:

1. What proportion of customers churn?
2. Which contract types have the highest observed churn rates?
3. How does customer tenure relate to churn?
4. Are higher monthly charges associated with higher churn?
5. Which payment methods are associated with higher churn?
6. Which internet services are associated with higher churn?
7. Which customer segments should receive retention attention?
8. Which features are most strongly associated with the model's churn predictions?
9. Which machine learning model provides the strongest predictive performance?

---

## 📈 Exploratory Data Analysis

The project includes analysis of:

* Overall churn distribution
* Churn by contract type
* Tenure distribution by churn status
* Churn by monthly-charge bands
* Churn by payment method
* Churn by internet service

### Key Observations

The analysis shows higher observed churn rates among:

* **Month-to-month customers**
* **Newer customers**
* Customers with **higher monthly charges**
* Customers using **electronic check**
* Customers using **fiber-optic internet service**

These findings identify customer segments that may warrant further investigation and targeted retention efforts.

> These relationships are observational associations in the analyzed dataset and should not be interpreted as evidence of causation.

---

## 🤖 Machine Learning

Two classification models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest**

The models were selected to provide both:

* An interpretable linear baseline
* A nonlinear tree-based model for comparison

### Train/Test Split

The dataset was divided using an **80/20 stratified train/test split**.

A fixed random seed was used:

```python
RANDOM_STATE = 42
```

Stratification helps maintain a similar churn-class distribution between the training and testing datasets.

---

## ⚙️ Preprocessing Pipeline

Preprocessing is implemented inside the machine learning pipeline to reduce the risk of data leakage.

### Numerical Features

* Median imputation
* Standard scaling

### Categorical Features

* Most-frequent imputation
* One-hot encoding
* `handle_unknown="ignore"`

The preprocessing steps are fitted using the training data and then applied to the test data.

Class weighting is used to help address class imbalance.

---

## 📊 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* ROC curves

**ROC-AUC is used as the primary model-selection metric.**

Recall is also important because missing customers who are likely to churn may reduce the effectiveness of a retention campaign.

### Model Performance

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Logistic Regression** | **0.743** | **0.510** | **0.783** | **0.618** | **0.849** |
| Random Forest           |     0.779 |     0.571 |     0.663 |     0.614 |     0.838 |

### Best Model

**Logistic Regression** achieved the highest ROC-AUC:

**ROC-AUC = 0.849**

Random Forest achieved higher accuracy and precision, but Logistic Regression achieved stronger ROC-AUC and recall.

Because ROC-AUC is the primary selection metric and recall is important for identifying potential churners, **Logistic Regression was selected as the best-performing model for this project.**

---

## 🧠 Model Interpretation

The selected Logistic Regression model provides an interpretable view of the relationship between features and predicted churn risk.

Model coefficients are analyzed to identify features with stronger positive or negative associations with churn predictions.

* A **positive coefficient** indicates an association with higher predicted churn probability.
* A **negative coefficient** indicates an association with lower predicted churn probability.
* Larger absolute coefficients indicate stronger model associations, subject to the feature encoding and scaling used in the pipeline.

Important features identified through the model interpretation include characteristics related to:

* Customer tenure
* Dependents
* Contract type
* Internet service
* Monthly charges
* Total charges
* Paperless billing

> Model coefficients represent statistical associations used by the predictive model and should not be interpreted as proof that a feature causes customer churn.

---

## 💡 Business Insights and Recommendations

The analysis suggests several customer segments that may deserve targeted retention attention.

### 1. New Customers

Customers with shorter tenure show higher observed churn rates.

**Potential action:**

Develop stronger onboarding, early engagement, and proactive support programs during the first months of service.

---

### 2. Month-to-Month Customers

Month-to-month customers represent an important higher-churn segment.

**Potential action:**

Investigate whether contract incentives, loyalty benefits, or improved long-term value propositions could encourage customers to consider longer-term contracts.

---

### 3. Higher Monthly Charges

Higher monthly-charge segments show elevated observed churn.

**Potential action:**

Investigate pricing, perceived value, service quality, and customer satisfaction among higher-paying customers.

---

### 4. Electronic Check Users

Customers using electronic check show higher observed churn.

**Potential action:**

Investigate whether payment experience or payment-method characteristics are associated with retention and test whether encouraging convenient automatic payment options improves retention.

---

### 5. Fiber-Optic Customers

Fiber-optic customers show elevated observed churn in the analyzed dataset.

**Potential action:**

Investigate service quality, technical issues, customer expectations, and perceived value before implementing targeted retention interventions.

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
│   └── churn_model.ipynb
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
│       ├── random_forest_confusion_matrix.png
│       └── logistic_regression_feature_importance.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📦 Project Outputs

The `outputs/` directory contains reusable analytical and modeling results:

* `best_churn_model.joblib` — saved best-performing churn model pipeline
* `model_comparison.csv` — model evaluation results
* `feature_importance.csv` — ranked Logistic Regression feature coefficients/importance information
* `eda.png` — exploratory data analysis visualizations
* `tenure_by_churn.png` — tenure analysis by churn status
* `business_churn_drivers.png` — business-oriented churn analysis
* `roc_curves.png` — ROC curve comparison
* `logistic_regression_confusion_matrix.png` — Logistic Regression confusion matrix
* `random_forest_confusion_matrix.png` — Random Forest confusion matrix
* `logistic_regression_feature_importance.png` — Logistic Regression coefficient visualization

---

## 🛠️ Technologies

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Development and Version Control

* Jupyter Notebook
* Git
* GitHub
* GitHub Codespaces

---

## 🚀 Future Improvements

Potential future improvements include:

* Probability threshold optimization
* Cost-sensitive decision analysis
* Customer risk segmentation
* Hyperparameter tuning
* Model calibration
* Model monitoring
* Data drift detection
* Production API deployment
* Integration with customer retention workflows

---

## ⚠️ Limitations

Several limitations should be considered:

* The analysis is based on a single historical dataset.
* Model performance is evaluated using a single held-out test split.
* Observed associations do not establish causation.
* Customer behavior and churn patterns may change over time.
* Model performance may change under future data distributions.
* False positives and false negatives have different business costs.
* The analysis does not estimate the financial cost or return of individual retention interventions.

---

## 👩‍💻 Author

**Javaria Qadeer**

Data Science | Machine Learning | Data Analytics

