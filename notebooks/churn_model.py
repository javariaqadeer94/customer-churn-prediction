"""IBM Telco Customer Churn: reproducible end-to-end workflow."""
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
							 precision_score, recall_score, roc_auc_score,
							 classification_report, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIGURES = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def main():
	csvs = sorted(RAW.glob("*.csv"))
	if not csvs:
		raise FileNotFoundError(f"No CSV dataset found in {RAW}")
	path = csvs[0]
	df = pd.read_csv(path)
	print(f"Loaded {path.relative_to(ROOT)} | shape={df.shape}")
	print("\nColumns:\n", df.columns.tolist())
	print("\nData types:\n", df.dtypes)
	print("\nBasic statistics:\n", df.describe(include="all").transpose())
	print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))

	df.columns = df.columns.str.strip()
	if "TotalCharges" in df:
		df["TotalCharges"] = pd.to_numeric(
			df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
		)
	if "customerID" in df:
		df = df.drop(columns="customerID")
	if "Churn" not in df:
		raise ValueError("Expected a Churn column")
	df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
	print("\nMissing values after cleaning (remaining values are imputed in pipelines):\n",
		  df.isna().sum().sort_values(ascending=False))

	# Simple EDA: class balance, contract relationship, and tenure relationship.
	fig, axes = plt.subplots(1, 2, figsize=(13, 4))
	sns.countplot(data=df, x="Churn", ax=axes[0])
	axes[0].set_title("Churn distribution")
	if "Contract" in df:
		sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[1])
		axes[1].tick_params(axis="x", rotation=20)
		axes[1].set_title("Churn by contract")
	plt.tight_layout(); fig.savefig(FIGURES / "eda.png", dpi=150); plt.close(fig)
	if "tenure" in df:
		fig = plt.figure(figsize=(7, 4))
		sns.boxplot(data=df, x="Churn", y="tenure")
		plt.title("Tenure by churn status")
		plt.tight_layout(); fig.savefig(FIGURES / "tenure_by_churn.png", dpi=150); plt.close(fig)

	X, y = df.drop(columns="Churn"), df["Churn"]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	numeric = X.select_dtypes(include=np.number).columns.tolist()
	categorical = X.select_dtypes(exclude=np.number).columns.tolist()
	prep = ColumnTransformer([
		("num", Pipeline([("impute", SimpleImputer(strategy="median")),
						  ("scale", StandardScaler())]), numeric),
		("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
						  ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
	])
	estimators = {
		"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
		"Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced",
												  random_state=42, n_jobs=-1),
	}
	fitted, rows = {}, []
	for name, estimator in estimators.items():
		model = Pipeline([("preprocessor", prep), ("model", estimator)])
		model.fit(X_train, y_train)  # preprocessing learns only from training data
		pred = model.predict(X_test)
		prob = model.predict_proba(X_test)[:, 1]
		rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
					 "Precision": precision_score(y_test, pred, zero_division=0),
					 "Recall": recall_score(y_test, pred, zero_division=0),
					 "F1-score": f1_score(y_test, pred, zero_division=0),
					 "ROC-AUC": roc_auc_score(y_test, prob)})
		fitted[name] = (model, prob)
		print(f"\n{name}\n", classification_report(y_test, pred, zero_division=0))
		print("Confusion matrix:\n", confusion_matrix(y_test, pred))
	comparison = pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
	print("\nModel comparison:\n", comparison.round(3))
	comparison.to_csv(OUT / "model_comparison.csv")
	best_name = comparison.index[0]
	best_model = fitted[best_name][0]
	joblib.dump(best_model, OUT / "best_churn_model.joblib")

	fig, ax = plt.subplots(figsize=(7, 5))
	for name, (model, prob) in fitted.items():
		RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax)
	ax.set_title("ROC curves"); plt.tight_layout(); fig.savefig(FIGURES / "roc_curves.png", dpi=150); plt.close(fig)
	names = best_model.named_steps["preprocessor"].get_feature_names_out()
	estimator = best_model.named_steps["model"]
	values = estimator.coef_[0] if hasattr(estimator, "coef_") else estimator.feature_importances_
	importance = pd.DataFrame({"Feature": names, "Importance": values,
							   "Absolute importance": np.abs(values)}).sort_values("Absolute importance", ascending=False)
	importance.head(20).to_csv(OUT / "feature_importance.csv", index=False)
	print("\nMost important features:\n", importance.head(10))

	readme = f"""# Customer Churn Prediction

End-to-end IBM Telco Customer Churn classification project. The script discovers the CSV in `data/raw/` without hardcoding its filename and never modifies the raw data.

## Methodology
Cleaning (`TotalCharges`, missing values, and `customerID`), EDA, stratified 80/20 split, one-hot encoding, scaling, and leakage-safe `ColumnTransformer` pipelines. Class-weighted Logistic Regression and Random Forest are compared using Accuracy, Precision, Recall, F1-score, confusion matrices, and ROC-AUC.

## Results
Best model by ROC-AUC: **{best_name}**

{comparison.round(3).to_markdown()}

## Recommendations
1. Prioritize retention outreach for short-tenure customers.
2. Offer flexible-plan incentives to month-to-month customers.
3. Review high monthly-charge accounts for value concerns.
4. Target limited retention budgets using model churn probabilities.
5. Improve onboarding and early-life customer support.
6. Monitor service, payment, and support segments for recurring churn patterns.

## Run
Install `pandas numpy matplotlib seaborn scikit-learn joblib tabulate`, place the unmodified CSV in `data/raw/`, then run `python notebooks/churn_model.py`. Outputs are saved in `outputs/`.
"""
	(ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
	main()
