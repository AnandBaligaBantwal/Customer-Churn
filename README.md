# Customer Churn Analysis & Segmentation

This project provides an end-to-end machine learning pipeline to analyze, predict, and segment telecom customer churn. Utilizing a combination of SQL-based data aggregation, predictive modeling with Random Forests, model explainability (SHAP & ELI5), and customer segmentation via K-Means clustering, this repository turns raw customer data into actionable business strategies.

## 🚀 Features

* **SQL-in-Python Aggregation:** Uses an in-memory SQLite database to dynamically engineer features like average tenure and internet usage metrics.
* **Predictive Modeling:** Implements a robust `scikit-learn Pipeline` utilizing a `RandomForestClassifier` with standardized numerical processing and one-hot encoded categorical variables.
* **Advanced Model Explainability:**
    * **ELI5 Permutation Importance:** Identifies which features have the biggest impact on model performance during testing.
    * **SHAP (SHapley Additive exPlanations):** Visualizes how individual feature values drive predictions up or down.
* **Customer Segmentation:** Groups customers into three distinct behavioral segments (**At Risk**, **Loyal**, and **Dormant**) using K-Means clustering to help tailor targeted retention campaigns.

---

## 🛠️ Project Structure

```text
├── CustChurn.py                       # Main Python script containing the pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Raw Telco Customer Churn dataset
├── processed_data.csv                 # Cleaned dataset with SQL engineered features
├── segmented_customers.csv             # Final dataset with assigned customer segments
├── churn_distribution.png             # Visual distribution of the target variable
├── correlation_matrix.png             # Correlation heatmap of numerical features
├── shap_summary_plot.png              # SHAP summary plot for global explainability
└── eli5_feature_importance.html       # Exported HTML file for permutation importance
