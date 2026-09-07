import pandas as pd
from pathlib import Path


# ==================================================
# FILES
# ==================================================

INPUT_FILE = Path(r"firms_osm_context_v1.csv")
OUTPUT_FILE = "firms_osm_features_v1.csv"


# ==================================================
# LOAD
# ==================================================

print("\n========== LOADING ==========")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)


# ==================================================
# INDUSTRIAL PROXIMITY
# ==================================================

print("\n========== INDUSTRIAL PROXIMITY ==========")

df["industrial_within_500m"] = (
    df["nearest_industrial_distance_m"] <= 500
).astype(int)

df["industrial_within_1km"] = (
    df["nearest_industrial_distance_m"] <= 1000
).astype(int)

df["industrial_within_5km"] = (
    df["nearest_industrial_distance_m"] <= 5000
).astype(int)


# ==================================================
# QUARRY PROXIMITY
# ==================================================

print("\n========== QUARRY PROXIMITY ==========")

df["quarry_within_500m"] = (
    df["nearest_quarry_distance_m"] <= 500
).astype(int)

df["quarry_within_1km"] = (
    df["nearest_quarry_distance_m"] <= 1000
).astype(int)

df["quarry_within_5km"] = (
    df["nearest_quarry_distance_m"] <= 5000
).astype(int)


# ==================================================
# CHECK DISTRIBUTIONS
# ==================================================

print("\n========== INDUSTRIAL FEATURES ==========")

for col in [
    "industrial_within_500m",
    "industrial_within_1km",
    "industrial_within_5km"
]:
    print(
        f"{col}:",
        df[col].sum(),
        f"({df[col].mean() * 100:.2f}%)"
    )


print("\n========== QUARRY FEATURES ==========")

for col in [
    "quarry_within_500m",
    "quarry_within_1km",
    "quarry_within_5km"
]:
    print(
        f"{col}:",
        df[col].sum(),
        f"({df[col].mean() * 100:.2f}%)"
    )


# ==================================================
# VALIDATION
# ==================================================

print("\n========== VALIDATION ==========")

print("Output shape:", df.shape)

print(
    "Unique source IDs:",
    df["source_id"].nunique()
)

print(
    "Duplicate source IDs:",
    df["source_id"].duplicated().sum()
)

print(
    "Missing values:",
    df.isna().sum().sum()
)


# ==================================================
# SAVE
# ==================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== SAVED ==========")

print(
    "Output:",
    OUTPUT_FILE
)