import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 1. LOAD DATA
# ============================================================

X = pd.read_csv("X_model_v1.csv")
y = pd.read_csv("y_model_v1.csv")["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 2. CREATE SPATIAL GROUPS
# ============================================================

GRID_SIZE = 0.5  # approximately 0.5° × 0.5° spatial cells

X["spatial_group"] = (
    np.floor(X["lat_grid"] / GRID_SIZE).astype(int).astype(str)
    + "_"
    + np.floor(X["lon_grid"] / GRID_SIZE).astype(int).astype(str)
)

print("\nNumber of spatial groups:", X["spatial_group"].nunique())


# ============================================================
# 3. ANALYZE POSITIVE-CLASS DISTRIBUTION ACROSS ALL GROUPS
# ============================================================

group_summary = (
    pd.DataFrame({
        "spatial_group": X["spatial_group"],
        "y": y
    })
    .groupby("spatial_group")
    .agg(
        total_sources=("y", "size"),
        industrial_sources=("y", "sum")
    )
    .reset_index()
)

group_summary["industrial_percent"] = (
    group_summary["industrial_sources"]
    / group_summary["total_sources"]
    * 100
)

group_summary["has_industrial"] = (
    group_summary["industrial_sources"] > 0
)


# ============================================================
# 4. PRINT OVERALL GROUP DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("POSITIVE-CLASS DISTRIBUTION ACROSS SPATIAL GROUPS")
print("=" * 60)

print(
    "Groups with at least one industrial source:",
    group_summary["has_industrial"].sum()
)

print(
    "Groups with zero industrial sources:",
    (~group_summary["has_industrial"]).sum()
)

print(
    "Industrial sources per group — minimum:",
    group_summary["industrial_sources"].min()
)

print(
    "Industrial sources per group — median:",
    group_summary["industrial_sources"].median()
)

print(
    "Industrial sources per group — mean:",
    round(group_summary["industrial_sources"].mean(), 3)
)

print(
    "Industrial sources per group — maximum:",
    group_summary["industrial_sources"].max()
)


# ============================================================
# 5. SHOW PERCENTILE DISTRIBUTION OF INDUSTRIAL COUNTS
# ============================================================

print("\nIndustrial sources per spatial group:")

print(
    group_summary["industrial_sources"]
    .quantile([0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
)


# ============================================================
# 6. SHOW TOP GROUPS BY INDUSTRIAL COUNT
# ============================================================

print("\nTop 20 spatial groups by industrial-source count:")

print(
    group_summary
    .sort_values("industrial_sources", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 7. SPATIAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# First:
#   70% train
#   30% temporary set
#
# Then split the temporary 30% equally:
#   15% validation
#   15% test
#
# Entire spatial groups remain together.
# ============================================================

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    first_split.split(
        X,
        y,
        groups=X["spatial_group"]
    )
)


# Split remaining 30% into 15% validation + 15% test

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    second_split.split(
        X_temp,
        y_temp,
        groups=X_temp["spatial_group"]
    )
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]


# ============================================================
# 8. CREATE FINAL SETS
# ============================================================

X_train = X.iloc[train_idx].drop(columns=["spatial_group"])
X_val = X.iloc[val_idx].drop(columns=["spatial_group"])
X_test = X.iloc[test_idx].drop(columns=["spatial_group"])

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]


# ============================================================
# 9. DISPLAY SET SIZES
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL SPLIT RESULTS")
print("=" * 60)

print("\nTRAIN")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nVALIDATION")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTEST")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(name, labels):

    counts = labels.value_counts().sort_index()
    percentages = labels.value_counts(normalize=True).sort_index() * 100

    print(f"\n{name} class distribution:")

    for cls in [0, 1]:
        print(
            f"  y={cls}: "
            f"{counts.get(cls, 0):,} "
            f"({percentages.get(cls, 0):.3f}%)"
        )


print_class_distribution("TRAIN", y_train)
print_class_distribution("VALIDATION", y_val)
print_class_distribution("TEST", y_test)


# ============================================================
# 11. ASSIGN FOLD TO EACH SPATIAL GROUP
# ============================================================

train_groups = set(
    X.iloc[train_idx]["spatial_group"]
)

val_groups = set(
    X.iloc[val_idx]["spatial_group"]
)

test_groups = set(
    X.iloc[test_idx]["spatial_group"]
)

group_summary["fold"] = group_summary["spatial_group"].apply(
    lambda g:
        "train" if g in train_groups
        else "validation" if g in val_groups
        else "test"
)


# ============================================================
# 12. CHECK POSITIVE CONCENTRATION BY FOLD
# ============================================================

fold_group_summary = (
    group_summary
    .groupby("fold")
    .agg(
        spatial_groups=("spatial_group", "count"),
        groups_with_industrial=("has_industrial", "sum"),
        total_sources=("total_sources", "sum"),
        industrial_sources=("industrial_sources", "sum")
    )
    .reset_index()
)

fold_group_summary["industrial_percent"] = (
    fold_group_summary["industrial_sources"]
    / fold_group_summary["total_sources"]
    * 100
)

print("\n" + "=" * 60)
print("SPATIAL-GROUP SUMMARY BY FOLD")
print("=" * 60)

print(
    fold_group_summary.to_string(index=False)
)


# ============================================================
# 13. SAVE SPATIAL GROUP ANALYSIS
# ============================================================

group_summary.to_csv(
    "spatial_group_summary_v1.csv",
    index=False
)


# ============================================================
# 14. SAVE TRAIN / VALIDATION / TEST
# ============================================================

X_train.to_csv("X_train_v2.csv", index=False)
X_val.to_csv("X_val_v2.csv", index=False)
X_test.to_csv("X_test_v2.csv", index=False)

y_train.to_csv("y_train_v2.csv", index=False)
y_val.to_csv("y_val_v2.csv", index=False)
y_test.to_csv("y_test_v2.csv", index=False)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\nSaved:")
print("X_train_v2.csv")
print("X_val_v2.csv")
print("X_test_v2.csv")
print("y_train_v2.csv")
print("y_val_v2.csv")
print("y_test_v2.csv")
print("spatial_group_summary_v1.csv")