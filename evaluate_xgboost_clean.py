import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=== FINAL TEST EVALUATION ===")

# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = pd.read_csv("X_test_clean_v1.csv")
y_test = pd.read_csv("y_test_v2.csv").squeeze()

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# --------------------------------------------------
# 2. Load the already-trained model
# --------------------------------------------------

model = xgb.XGBClassifier()

model.load_model("xgboost_model_clean_v1.json")

print("Model loaded successfully.")

# --------------------------------------------------
# 3. Generate test predictions
# --------------------------------------------------

y_test_probability = model.predict_proba(X_test)[:, 1]

y_test_prediction = (y_test_probability >= 0.5).astype(int)

# --------------------------------------------------
# 4. Calculate final metrics
# --------------------------------------------------

pr_auc = average_precision_score(
    y_test,
    y_test_probability
)

roc_auc = roc_auc_score(
    y_test,
    y_test_probability
)

cm = confusion_matrix(
    y_test,
    y_test_prediction
)

print("\n=== FINAL TEST RESULTS ===")

print("\nPR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_prediction,
        digits=4,
        zero_division=0
    )
)

print("\n=== TEST EVALUATION COMPLETE ===")