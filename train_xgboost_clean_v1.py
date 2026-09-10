import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=== CLEAN XGBOOST TRAINING ===")

# --------------------------------------------------
# 1. Load clean spatial-split datasets
# --------------------------------------------------

X_train = pd.read_csv("X_train_clean_v1.csv")
X_val = pd.read_csv("X_val_clean_v1.csv")

y_train = pd.read_csv("y_train_v2.csv").squeeze()
y_val = pd.read_csv("y_val_v2.csv").squeeze()

print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


# --------------------------------------------------
# 2. Check class distribution
# --------------------------------------------------

train_neg = (y_train == 0).sum()
train_pos = (y_train == 1).sum()

print("\nTraining class distribution:")
print("Negative:", train_neg)
print("Positive:", train_pos)

scale_pos_weight = train_neg / train_pos

print("scale_pos_weight:", scale_pos_weight)


# --------------------------------------------------
# 3. Train XGBoost
# --------------------------------------------------

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nStarting training...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print("\nTraining complete.")


# --------------------------------------------------
# 4. Validation predictions
# --------------------------------------------------

y_val_probability = model.predict_proba(X_val)[:, 1]

y_val_prediction = (y_val_probability >= 0.5).astype(int)


# --------------------------------------------------
# 5. Validation metrics
# --------------------------------------------------

pr_auc = average_precision_score(
    y_val,
    y_val_probability
)

roc_auc = roc_auc_score(
    y_val,
    y_val_probability
)

cm = confusion_matrix(
    y_val,
    y_val_prediction
)

print("\n=== VALIDATION RESULTS ===")

print("\nPR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_val_prediction,
        digits=4,
        zero_division=0
    )
)


# --------------------------------------------------
# 6. Save model
# --------------------------------------------------

model.save_model("xgboost_model_clean_v1.json")

print("\nModel saved:")
print("xgboost_model_clean_v1.json")

print("\n=== COMPLETE ===")