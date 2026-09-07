import pandas as pd
from pathlib import Path

# ==================================================
# LOAD FIRMS SOURCE FEATURES
# ==================================================

FIRMS_FILE = Path(r"Dataset\FIRMS_features\firms_source_features_v1.csv")
firms = pd.read_csv(FIRMS_FILE)

print("\n========== BASIC INFO ==========")
print("Shape:", firms.shape)

print("\n========== COLUMNS ==========")
print(firms.columns.tolist())

# ==================================================
# COORDINATES
# ==================================================

print("\n========== FIRMS SOURCE COORDINATES ==========")

print(
    firms[
        ["lat_grid", "lon_grid"]
    ].describe()
)

print("\n========== MISSING COORDINATES ==========")

print(
    firms[
        ["lat_grid", "lon_grid"]
    ].isna().sum()
)

# ==================================================
# DUPLICATE SOURCE COORDINATES
# ==================================================

print("\n========== DUPLICATE SOURCE COORDINATES ==========")

duplicate_coordinates = firms.duplicated(
    subset=["lat_grid", "lon_grid"]
).sum()

print(
    "Duplicate (lat_grid, lon_grid):",
    duplicate_coordinates
)

# ==================================================
# SOURCE ID
# ==================================================

print("\n========== SOURCE ID ==========")

print(
    "Unique source IDs:",
    firms["source_id"].nunique()
)

print(
    "Duplicate source IDs:",
    firms["source_id"].duplicated().sum()
)

# ==================================================
# INDIA RANGE CHECK
# ==================================================

print("\n========== INDIA RANGE CHECK ==========")

outside_india_bbox = (
    (firms["lat_grid"] < 8) |
    (firms["lat_grid"] > 37) |
    (firms["lon_grid"] < 67) |
    (firms["lon_grid"] > 98)
)

print(
    "Sources outside rough India bbox:",
    outside_india_bbox.sum()
)

# ==================================================
# SAMPLE
# ==================================================

print("\n========== SAMPLE SOURCES ==========")

print(
    firms[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "detection_count",
            "detection_days",
            "persistence_7d_max",
            "persistence_30d_max",
            "persistence_90d_max"
        ]
    ]
    .head(20)
    .to_string(index=False)
)