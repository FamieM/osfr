import pandas as pd

print("=== CREATING CLEAN OSM FEATURES ===")

# --------------------------------------------------
# 1. Load the already-created spatial splits
# --------------------------------------------------

X_train = pd.read_csv("X_train_encoded_v1.csv")
X_val = pd.read_csv("X_val_encoded_v1.csv")
X_test = pd.read_csv("X_test_encoded_v1.csv")

print("Original shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# --------------------------------------------------
# 2. Columns that must NOT be used as predictors
# --------------------------------------------------
# These are FIRMS-derived features.
# The labels themselves were created from FIRMS behavior,
# so we remove the FIRMS behavioral/thermal variables
# to avoid target leakage.

firms_columns = [
    "detection_count",
    "mean_frp",
    "mean_brightness",
    "max_brightness",
    "mean_thermal_delta",
    "max_thermal_delta",
    "persistence_7d_max",
    "persistence_30d_max",
    "persistence_90d_max",
    "night_ratio",
    "type_0_ratio",
    "type_2_ratio",
    "type_3_ratio"
]


# --------------------------------------------------
# 3. Identifier columns
# --------------------------------------------------

identifier_columns = [
    "source_id",
    "nearest_industrial_osm_id",
    "nearest_quarry_osm_id"
]


# --------------------------------------------------
# 4. Raw date columns
# --------------------------------------------------

date_columns = [
    "first_seen",
    "last_seen"
]


# --------------------------------------------------
# 5. Columns to remove
# --------------------------------------------------

columns_to_remove = (
    firms_columns
    + identifier_columns
    + date_columns
)

print("\nRemoving columns:")
for col in columns_to_remove:
    if col in X_train.columns:
        print("-", col)


# --------------------------------------------------
# 6. Create clean feature sets
# --------------------------------------------------

X_train_clean = X_train.drop(
    columns=columns_to_remove,
    errors="ignore"
)

X_val_clean = X_val.drop(
    columns=columns_to_remove,
    errors="ignore"
)

X_test_clean = X_test.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# --------------------------------------------------
# 7. Verify identical feature columns
# --------------------------------------------------

assert list(X_train_clean.columns) == list(X_val_clean.columns)
assert list(X_train_clean.columns) == list(X_test_clean.columns)

print("\nClean shapes:")
print("Train:", X_train_clean.shape)
print("Validation:", X_val_clean.shape)
print("Test:", X_test_clean.shape)

print("\nRemaining features:")
for col in X_train_clean.columns:
    print(col)


# --------------------------------------------------
# 8. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print("Train:", int(X_train_clean.isna().sum().sum()))
print("Validation:", int(X_val_clean.isna().sum().sum()))
print("Test:", int(X_test_clean.isna().sum().sum()))


# --------------------------------------------------
# 9. Save clean datasets
# --------------------------------------------------

X_train_clean.to_csv("X_train_clean_v1.csv", index=False)
X_val_clean.to_csv("X_val_clean_v1.csv", index=False)
X_test_clean.to_csv("X_test_clean_v1.csv", index=False)

print("\nSaved:")
print("X_train_clean_v1.csv")
print("X_val_clean_v1.csv")
print("X_test_clean_v1.csv")

print("\n=== CLEAN FEATURE CREATION COMPLETE ===")