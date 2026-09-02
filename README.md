# Telco Customer Churn Prediction

An end-to-end data analytics and machine learning project focused on understanding customer churn and predicting customers at risk of leaving a telecommunications service.

## 📌 Project Overview

Customer churn is an important business problem for telecommunications companies because losing customers can directly affect revenue and customer lifetime value.

This project analyzes customer demographics, services, account information, contract characteristics, and billing behavior to identify patterns associated with customer churn.

The project combines **data analytics, exploratory data analysis, visualization, feature engineering, and machine learning** to develop an interpretable customer churn prediction workflow.

### Project Workflow

**Business Understanding → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Machine Learning → Model Evaluation → Explainability → Business Insights**

---

## 🎯 Objectives

* Understand the characteristics of customers who churn.
* Identify key factors associated with customer churn.
* Clean and prepare customer data for analysis.
* Perform exploratory data analysis and visualization.
* Develop meaningful features for predictive modeling.
* Train and compare multiple classification models.
* Evaluate models using appropriate business-focused metrics.
* Identify important drivers of churn.
* Develop a reusable churn prediction pipeline.
* Explore deployment of the model through an interactive application.

---

## 📊 Dataset

The project uses a Telco Customer Churn dataset obtained from Kaggle.

The dataset contains customer-level information including:

* Demographics
* Customer tenure
* Contract type
* Internet services
* Additional services
* Payment method
* Monthly charges
* Total charges
* Churn status

### Target Variable

`Churn`

* `Yes` — Customer churned
* `No` — Customer did not churn

The original dataset is kept separately from the analysis code where required by the dataset's licensing and redistribution conditions.

---

## 🔍 Business Questions

This project investigates questions such as:

1. What percentage of customers churn?
2. Which customer segments have higher churn rates?
3. Does contract type influence churn?
4. How does customer tenure relate to churn?
5. Are customers with higher monthly charges more likely to churn?
6. Which services are associated with customer retention or churn?
7. Which payment methods are associated with higher churn?
8. Which features are most useful for predicting churn?
9. Which machine learning model provides the best predictive performance?

---

## 📈 Data Analytics

The exploratory analysis will examine:

* Overall churn distribution
* Churn by contract type
* Churn by tenure
* Churn by monthly charges
* Churn by payment method
* Churn by internet service
* Customer service subscriptions
* Demographic characteristics
* Relationships between numerical variables and churn

Visualizations will be used to communicate the most important findings clearly.

---

## 🤖 Machine Learning

The project will compare several classification algorithms, including:

* Logistic Regression
* Random Forest
* XGBoost

Model performance will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Because the business objective is to identify customers who may leave, particular attention will be given to **recall for the churn class** and the precision-recall trade-off.

---

## 🧠 Model Explainability

Model explainability techniques will be used to understand which customer characteristics contribute most strongly to churn predictions.

This helps move the project beyond simply predicting churn toward generating insights that could support customer retention strategies.

---

## 📁 Project Structure

```text
telco-customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_telco_churn_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── app/
│   └── app.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🛠️ Technologies

**Programming**

* Python

**Data Analysis**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Machine Learning**

* Scikit-learn
* XGBoost

**Explainability**

* SHAP

**Development**

* Jupyter Notebook
* Git
* GitHub

**Deployment**

* Streamlit

---

## 📊 Model Performance

Results will be added after completing model training and evaluation.

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |        — |         — |      — |        — |       — |
| Random Forest       |        — |         — |      — |        — |       — |
| XGBoost             |        — |         — |      — |        — |       — |

---

## 💡 Business Insights

The final analysis will translate model and exploratory findings into actionable business insights.

The analysis will focus on identifying:

* High-risk customer segments
* Contract-related churn patterns
* Tenure-related churn patterns
* Billing-related churn patterns
* Service-related churn patterns
* Factors that may support customer retention strategies

Final insights will be based on the actual results obtained from the analysis.

---

## 🚀 Future Improvements

Potential improvements include:

* Cost-sensitive learning
* Probability threshold optimization
* Customer risk segmentation
* Model monitoring
* Data drift detection
* Production API deployment
* Integration with customer retention workflows

---

## 👩‍💻 Author

**Javaria Qadeer**

Data Science | Machine Learning | Data Analytics

This project is part of my professional portfolio development toward Data Analyst and Data Scientist roles.
