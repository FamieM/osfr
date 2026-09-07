import pandas as pd
from pathlib import Path

# ==================================================
# LOAD
# ==================================================

FILE = Path(r"firms_osm_context_v1.csv")

df = pd.read_csv(FILE)

print("\n========== BASIC INFO ==========")

print("Shape:", df.shape)

# ==================================================
# INDUSTRIAL DISTANCE
# ==================================================

print("\n========== INDUSTRIAL DISTANCE ==========")

industrial = df["nearest_industrial_distance_m"]

print(industrial.describe())

print("\nIndustrial proximity thresholds:")

for threshold in [250, 500, 1000, 2000, 5000, 10000, 25000, 50000]:
    count = (industrial <= threshold).sum()
    percentage = count / len(df) * 100

    print(
        f"Within {threshold:>6,} m: "
        f"{count:>7,} "
        f"({percentage:6.2f}%)"
    )

# ==================================================
# QUARRY DISTANCE
# ==================================================

print("\n========== QUARRY DISTANCE ==========")

quarry = df["nearest_quarry_distance_m"]

print(quarry.describe())

print("\nQuarry proximity thresholds:")

for threshold in [250, 500, 1000, 2000, 5000, 10000, 25000, 50000]:
    count = (quarry <= threshold).sum()
    percentage = count / len(df) * 100

    print(
        f"Within {threshold:>6,} m: "
        f"{count:>7,} "
        f"({percentage:6.2f}%)"
    )

# ==================================================
# FACILITY TYPE
# ==================================================

print("\n========== NEAREST INDUSTRIAL FACILITY TYPE ==========")

print(
    df["nearest_industrial_facility_type"]
    .value_counts(dropna=False)
)

print("\n========== NEAREST QUARRY FACILITY TYPE ==========")

print(
    df["nearest_quarry_facility_type"]
    .value_counts(dropna=False)
)

# ==================================================
# STATE DISTRIBUTION
# ==================================================

print("\n========== NEAREST INDUSTRIAL STATE ==========")

print(
    df["nearest_industrial_state"]
    .value_counts()
    .head(20)
)

print("\n========== NEAREST QUARRY STATE ==========")

print(
    df["nearest_quarry_state"]
    .value_counts()
    .head(20)
)

# ==================================================
# CLOSEST FIRMS SOURCES
# ==================================================

print("\n========== CLOSEST TO INDUSTRIAL ==========")

print(
    df[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "detection_days",
            "persistence_30d_max",
            "mean_frp",
            "nearest_industrial_distance_m"
        ]
    ]
    .sort_values(
        "nearest_industrial_distance_m"
    )
    .head(20)
    .to_string(index=False)
)

print("\n========== CLOSEST TO QUARRY ==========")

print(
    df[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "detection_days",
            "persistence_30d_max",
            "mean_frp",
            "nearest_quarry_distance_m"
        ]
    ]
    .sort_values(
        "nearest_quarry_distance_m"
    )
    .head(20)
    .to_string(index=False)
)

# ==================================================
# HIGH PERSISTENCE + INDUSTRIAL PROXIMITY
# ==================================================

print("\n========== HIGH-PERSISTENCE SOURCES NEAR INDUSTRY ==========")

subset = df[
    (df["persistence_30d_max"] >= 5) &
    (df["nearest_industrial_distance_m"] <= 5000)
]

print(
    "Sources meeting both conditions:",
    len(subset)
)

print(
    subset[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "detection_days",
            "persistence_30d_max",
            "persistence_90d_max",
            "mean_frp",
            "max_frp",
            "nearest_industrial_distance_m"
        ]
    ]
    .sort_values(
        [
            "persistence_30d_max",
            "nearest_industrial_distance_m"
        ],
        ascending=[False, True]
    )
    .head(20)
    .to_string(index=False)
)

print("\n========== COMPLETE ==========")