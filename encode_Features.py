import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. LOAD SPATIAL TRAIN / VALIDATION / TEST DATA
# ============================================================

X_train = pd.read_csv("X_train_v2.csv")
X_val = pd.read_csv("X_val_v2.csv")
X_test = pd.read_csv("X_test_v2.csv")

y_train = pd.read_csv("y_train_v2.csv")["y"]
y_val = pd.read_csv("y_val_v2.csv")["y"]
y_test = pd.read_csv("y_test_v2.csv")["y"]


# ============================================================
# 2. CATEGORICAL FEATURES
# ============================================================

categorical_cols = [
    "nearest_industrial_state",
    "nearest_quarry_state"
]

print("Categorical features:")
for col in categorical_cols:
    print(" -", col)


# ============================================================
# 3. FIT ENCODER USING TRAIN ONLY
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoder.fit(X_train[categorical_cols])


# ============================================================
# 4. SHOW FROZEN TRAIN CATEGORIES
# ============================================================

print("\nCategories learned from TRAIN only:")

for col, categories in zip(
    categorical_cols,
    encoder.categories_
):
    print(f"\n{col}:")
    print(list(categories))
    print("Number of categories:", len(categories))


# ============================================================
# 5. ENCODE TRAIN / VALIDATION / TEST
# ============================================================

train_encoded = encoder.transform(
    X_train[categorical_cols]
)

val_encoded = encoder.transform(
    X_val[categorical_cols]
)

test_encoded = encoder.transform(
    X_test[categorical_cols]
)


# ============================================================
# 6. GET ENCODED COLUMN NAMES
# ============================================================

encoded_columns = encoder.get_feature_names_out(
    categorical_cols
)


# ============================================================
# 7. REMOVE ORIGINAL CATEGORICAL COLUMNS
# ============================================================

X_train_numeric = X_train.drop(
    columns=categorical_cols
)

X_val_numeric = X_val.drop(
    columns=categorical_cols
)

X_test_numeric = X_test.drop(
    columns=categorical_cols
)


# ============================================================
# 8. COMBINE NUMERIC + ONE-HOT FEATURES
# ============================================================

X_train_final = pd.concat(
    [
        X_train_numeric.reset_index(drop=True),
        pd.DataFrame(
            train_encoded,
            columns=encoded_columns
        )
    ],
    axis=1
)

X_val_final = pd.concat(
    [
        X_val_numeric.reset_index(drop=True),
        pd.DataFrame(
            val_encoded,
            columns=encoded_columns
        )
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numeric.reset_index(drop=True),
        pd.DataFrame(
            test_encoded,
            columns=encoded_columns
        )
    ],
    axis=1
)


# ============================================================
# 9. CHECK SHAPES
# ============================================================

print("\nFinal feature shapes:")

print("X_train_final:", X_train_final.shape)
print("X_val_final:", X_val_final.shape)
print("X_test_final:", X_test_final.shape)


# ============================================================
# 10. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")

print("Train:", X_train_final.isna().sum().sum())
print("Validation:", X_val_final.isna().sum().sum())
print("Test:", X_test_final.isna().sum().sum())


# ============================================================
# 11. CHECK UNKNOWN CATEGORIES
# ============================================================
#
# Because handle_unknown="ignore" is enabled, unseen
# validation/test categories become all-zero indicators.
# We report them here for transparency.
# ============================================================

for col in categorical_cols:

    train_categories = set(
        X_train[col].dropna().unique()
    )

    val_categories = set(
        X_val[col].dropna().unique()
    )

    test_categories = set(
        X_test[col].dropna().unique()
    )

    unseen_val = sorted(
        val_categories - train_categories
    )

    unseen_test = sorted(
        test_categories - train_categories
    )

    print(f"\n{col}")

    print(
        "Unseen in validation:",
        unseen_val
    )

    print(
        "Unseen in test:",
        unseen_test
    )


# ============================================================
# 12. SAVE FINAL FEATURES
# ============================================================

X_train_final.to_csv(
    "X_train_encoded_v1.csv",
    index=False
)

X_val_final.to_csv(
    "X_val_encoded_v1.csv",
    index=False
)

X_test_final.to_csv(
    "X_test_encoded_v1.csv",
    index=False
)


# ============================================================
# 13. SAVE TARGETS WITH MATCHING ROW ORDER
# ============================================================

y_train.to_csv(
    "y_train_encoded_v1.csv",
    index=False
)

y_val.to_csv(
    "y_val_encoded_v1.csv",
    index=False
)

y_test.to_csv(
    "y_test_encoded_v1.csv",
    index=False
)


print("\nSaved:")
print("X_train_encoded_v1.csv")
print("X_val_encoded_v1.csv")
print("X_test_encoded_v1.csv")
print("y_train_encoded_v1.csv")
print("y_val_encoded_v1.csv")
print("y_test_encoded_v1.csv")