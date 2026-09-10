import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

print("=== SHAP ANALYSIS ===")

# --------------------------------------------------
# 1. Load clean test data
# --------------------------------------------------

X_test = pd.read_csv("X_test_clean_v1.csv")

print("Test data:", X_test.shape)

# --------------------------------------------------
# 2. Load final trained model
# --------------------------------------------------

model = xgb.XGBClassifier()
model.load_model("xgboost_model_clean_v1.json")

print("Model loaded.")

# --------------------------------------------------
# 3. Use a sample for SHAP
# --------------------------------------------------
# Full SHAP calculation can be slow.
# 5,000 samples are enough for a presentation-level
# explanation.

X_sample = X_test.sample(
    n=min(5000, len(X_test)),
    random_state=42
)

print("SHAP sample:", X_sample.shape)

# --------------------------------------------------
# 4. Calculate SHAP values
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated.")

# --------------------------------------------------
# 5. Summary plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary_v1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("shap_summary_v1.png")

# --------------------------------------------------
# 6. Bar importance plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_feature_importance_v1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("shap_feature_importance_v1.png")

print("\n=== SHAP ANALYSIS COMPLETE ===")