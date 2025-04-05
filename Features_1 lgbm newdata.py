# Reload dataset and re-run full pipeline using shortened labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load dataset
# file_path = r"E:\UoB Year 3\Data Science\Final Table (from Statutory Datasets)(Combined Table).csv"
file_path = r"E:\UoB Year 3\Data Science\Data Analysis\Final_table_pca_0.csv"
df = pd.read_csv(file_path, encoding="latin1")
df = df.drop(columns=["Unnamed: 25"], errors="ignore")

# Define target and features
target_col = "Total number of households whose prevention duty ended with accommodation secured1 (P2)"
feature_cols = [
    # "Total PRS (A4P)",
    # "Total SRS (A4P)",
    "PRS_SRS_PC1",
    "Family_Friends_PC1",
    "Owner-occupier / shared ownership (A4P)",
    # "Living with family (A4P)",
    # "Living with friends (A4P)",
    "Temporary accommodation (A4P)",
    "No fixed abode3 (A4P)",
    "Rough sleeping (A4P)",
    "Refuge (A4P)",
    "National Asylum Seeker Support (NASS) accommodation (A4P)",
    "Total homeless on departure from institution (A4P)",
    "Other / not known4 (A4P)",
    "Accommodation secured by local authority or organisation delivering housing options service (P3)",
    "Helped to secure accommodation found by applicant, with financial payment (P3)",
    "Helped to secure accommodation found by applicant, without financial payment (P3)",
    "Supported housing provided (P3)",
    "Negotiation / mediation work to secure return to family or friend (P3)",
    "Negotiation / mediation / advocacy work to prevent eviction / repossession (P3)",
    "Discretionary Housing Payment to reduce shortfall (P3)",
    "Other financial payments (e.g. to reduce arrears)3 (P3)",
    "Other2 (P3)",
    "No activity \x96 advice and information provided (P3)",
]


# Function to shorten labels
def shorten_label(label, max_words=4):
    return " ".join(label.split()[:max_words])


# Prepare X and y
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target_col], errors="coerce")
Xy = pd.concat([X, y], axis=1).dropna()
X = Xy[feature_cols]
# X.columns = X.columns.str.replace(r"[^\w\s]", "", regex=True).str.replace(" ", " and ")
X.columns = X.columns.str.replace(r"[^\w\s]", "", regex=True).str.replace(
    "/", " and ", regex=False
)
shortened_labels = {col: shorten_label(col) for col in X.columns}
y = Xy[target_col]
X_short = X.rename(columns=shortened_labels)


# Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from lightgbm import LGBMRegressor

model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance
# Feature importance from LightGBM
# Fully reset the DataFrame from scratch using LightGBM model
feature_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Add the short label column for plotting
feature_importance_df["Short Feature"] = feature_importance_df["Feature"].map(
    shortened_labels
)

# Slice the top 10
top_features = feature_importance_df.head(10)
print(feature_importance_df.head(10))


# Plot feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     data=top_features, y="Short Feature", x="Importance", palette="coolwarm", ci=None
# )
# plt.title(
#     "10 Most Important Factors Determined by Decision Tree \n for Predicting Successful Homelessness Prevention"
# )
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig("top_10_feature_shortlabel.png", dpi=600)
# plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_features, y="Short Feature", x="Importance", palette="coolwarm", ci=None
)
plt.title("Top 10 Most Important Features Determined by LightGBM")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("top_10_feature_lgbm_short.png", dpi=600)
plt.show()


# Predicted vs actual
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_test, y=y_pred, alpha=0.7, color="dodgerblue", edgecolor="w", marker=".", s=20
)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="gray")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(
    f"Predicted vs Actual Number of Accommodation Secured \nRMSE: {rmse:.2f}, R²: {r2:.2f}"
)
plt.tight_layout()
plt.savefig("predicted_vs_actual_shortlabel.png", dpi=300)
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=y_pred, y=residuals, alpha=0.6, color="tomato", edgecolor="w", marker=".", s=50
)
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Predicted")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot_shortlabel.png", dpi=300)
plt.show()

# Save top 10 prediction examples
examples_df = X_test.copy()
examples_df["Actual"] = y_test
examples_df["Predicted"] = y_pred
examples_df["Residual"] = examples_df["Actual"] - examples_df["Predicted"]
examples_df = examples_df.reset_index(drop=True)
examples_df.head(10).to_csv("prediction_examples.csv", index=False)

# Save trained model
import joblib

joblib.dump(model, "gb_model_shortlabel.pkl")

# Interactive scatter plot
import plotly.express as px

df_plot = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
fig = px.scatter(
    df_plot,
    x="Actual",
    y="Predicted",
    trendline="ols",
    title="Predicted vs Actual Number of Accommodation Secured",
    color_discrete_sequence=["deepskyblue"],
    height=500,
)
fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
fig.write_html("predicted_vs_actual_interactive.html")
