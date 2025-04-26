import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sqlite3
import shap
import eli5
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from eli5.sklearn import PermutationImportance
from sklearn.cluster import KMeans

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#SQLite query
conn = sqlite3.connect(":memory:")
df.to_sql("telco", conn, index=False)

query = """
SELECT 
    customerID,
    AVG(tenure) as [AvgTenure],
    SUM(MonthlyCharges) as [TotalMonthlyCharges],
    COUNT(CASE WHEN InternetService != 'No' THEN 1 END) as [InternetUsage]
FROM telco
GROUP BY customerID
"""
agg_df = pd.read_sql(query, conn, index_col="customerID")


df = df.merge(agg_df, on="customerID", how="left")

df

df.info()
df.describe()
df.isnull().sum()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.to_csv("processed_data.csv", index=False)


#Countplot
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

df_numdata = ["tenure", "MonthlyCharges", "TotalCharges"]

#correlation matrix
plt.figure(figsize=(8, 6))
plt.title("Correlation Matrix")
sns.heatmap(
    df[df_numdata + ["Churn"]].replace({"Yes": 1, "No": 0}).corr(),
    annot=True,
    fmt=".2f",
)
plt.savefig("correlation_matrix.png")
plt.show()

#binary Classification Model
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
    ]
)

#Random Forest Classifier
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")


# Feature Importance using ELI5
feature_names = (
    numerical_cols.tolist()
    + pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
)

perm = PermutationImportance(pipeline.named_steps["classifier"], random_state=42).fit(
    pipeline.named_steps["preprocessor"].transform(X_test), y_test
)

eli5_html = eli5.show_weights(perm, feature_names=feature_names).data
with open("eli5_feature_importance.html", "w") as f:
    f.write(eli5_html)

# SHAP 
X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
shap_values = explainer.shap_values(X_test_transformed)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[:, :, 1],
    pipeline.named_steps["preprocessor"].transform(X_test),
    feature_names=feature_names,
    show=False,
)
plt.title("SHAP Summary Plot")
plt.savefig("shap_summary_plot.png")
plt.show()


cluster_features = ["tenure", "MonthlyCharges", "TotalCharges"]
X_cluster = df[cluster_features].copy()
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Segment"] = kmeans.fit_predict(X_cluster_scaled)

segment_labels = {0: "At Risk", 1: "Loyal", 2: "Dormant"}
df["Segment"] = df["Segment"].map(segment_labels)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="tenure", y="MonthlyCharges", hue="Segment", size="TotalCharges", data=df
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.title("Customer Segments")
plt.savefig("customer_segments.png")
plt.show()

segment_summary = df.groupby("Segment").agg(
    {
        "tenure": ["mean", "count"],
        "MonthlyCharges": "mean",
        "Churn": lambda x: (x == "Yes").mean(),
    }
)
print(segment_summary)


df.to_csv("segmented_customers.csv", index=False)
