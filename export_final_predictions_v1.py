import pandas as pd
import xgboost as xgb

print("=== EXPORTING FINAL TEST PREDICTIONS ===")

# Load test features and labels
X_test = pd.read_csv("X_test_clean_v1.csv")
y_test = pd.read_csv("y_test_v2.csv").squeeze()

# Load final trained model
model = xgb.XGBClassifier()
model.load_model("xgboost_model_clean_v1.json")

# Predict probability
y_probability = model.predict_proba(X_test)[:, 1]

# Classification at threshold 0.5
y_prediction = (y_probability >= 0.5).astype(int)

# Create output
predictions = X_test.copy()

predictions["actual_label"] = y_test.values
predictions["predicted_probability"] = y_probability
predictions["predicted_label"] = y_prediction

# Save
predictions.to_csv(
    "final_test_predictions_v1.csv",
    index=False
)

print("\nSaved:")
print("final_test_predictions_v1.csv")

print("\nPrediction counts:")
print(predictions["predicted_label"].value_counts().sort_index())

print("\n=== EXPORT COMPLETE ===")