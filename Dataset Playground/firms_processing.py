import pandas as pd
from pathlib import Path
import numpy as np

# --------------------------------------------------
# 1. LOAD RAW FIRMS DATA
# --------------------------------------------------

FILE_PATH = Path(__file__).parent / "Dataset" / "fire_archive_SV-C2_793789.csv"

firms = pd.read_csv(FILE_PATH)

print("Raw shape:", firms.shape)


# --------------------------------------------------
# 2. CONVERT DATE
# --------------------------------------------------

firms["acq_date"] = pd.to_datetime(firms["acq_date"])


# --------------------------------------------------
# 3. CREATE THERMAL FEATURE
# --------------------------------------------------

firms["thermal_delta"] = (
    firms["brightness"] - firms["bright_t31"]
)


# --------------------------------------------------
# 4. CREATE APPROXIMATE SPATIAL GRID
# --------------------------------------------------

firms["lat_grid"] = firms["latitude"].round(2)
firms["lon_grid"] = firms["longitude"].round(2)


# # --------------------------------------------------
# # 5. BASIC CHECK
# # --------------------------------------------------

# print("\nProcessed shape:", firms.shape)

# print("\nNew columns:")
# print(
#     firms[
#         [
#             "latitude",
#             "longitude",
#             "acq_date",
#             "brightness",
#             "bright_t31",
#             "thermal_delta",
#             "lat_grid",
#             "lon_grid"
#         ]
#     ].head()
# )

# print("\nUnique grid cells:",
#       firms[["lat_grid", "lon_grid"]].drop_duplicates().shape[0])
# --------------------------------------------------
# 6. BUILD CANDIDATE SOURCE TABLE
# --------------------------------------------------

source_table = (
    firms
    .groupby(["lat_grid", "lon_grid"])
    .agg(
        first_seen=("acq_date", "min"),
        last_seen=("acq_date", "max"),

        detection_count=("acq_date", "size"),
        detection_days=("acq_date", "nunique"),

        mean_frp=("frp", "mean"),
        max_frp=("frp", "max"),

        mean_brightness=("brightness", "mean"),
        max_brightness=("brightness", "max"),

        mean_thermal_delta=("thermal_delta", "mean"),
        max_thermal_delta=("thermal_delta", "max"),
    )
    .reset_index()
)


# --------------------------------------------------
# 7. CREATE SOURCE ID
# --------------------------------------------------

source_table.insert(
    0,
    "source_id",
    range(1, len(source_table) + 1)
)


# # --------------------------------------------------
# # 8. INSPECT RESULT
# # --------------------------------------------------

# print("\n=== SOURCE TABLE ===")

# print("Shape:", source_table.shape)

# print("\nColumns:")
# print(source_table.columns.tolist())

# print("\nFirst 10 sources:")
# print(source_table.head(10))

# print("\nDetection-day distribution:")
# print(
#     source_table["detection_days"].describe(
#         percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
#     )
# )
# --------------------------------------------------
# 9. UNIQUE DETECTION DAYS PER SOURCE
# --------------------------------------------------

source_dates = (
    firms[
        ["lat_grid", "lon_grid", "acq_date"]
    ]
    .drop_duplicates()
    .sort_values(
        ["lat_grid", "lon_grid", "acq_date"]
    )
)

print("\n=== SOURCE DATE TABLE ===")

print("Shape:", source_dates.shape)

print("\nFirst 10 rows:")
print(source_dates.head(10))

# --------------------------------------------------
# 10. CALCULATE 7-DAY PERSISTENCE
# --------------------------------------------------

# source_dates = source_dates.sort_values(
#     ["lat_grid", "lon_grid", "acq_date"]
# )


# def calculate_7d_persistence(group):
#     dates = group["acq_date"].values

#     result = []

#     for current_date in dates:
#         start_date = current_date - pd.Timedelta(days=6)

#         count = (
#             (dates >= start_date) &
#             (dates <= current_date)
#         ).sum()

#         result.append(count)

#     group = group.copy()
#     group["persistence_7d"] = result

#     return group


# source_dates_7d = (
#     source_dates
#     .groupby(["lat_grid", "lon_grid"], group_keys=False)
#     .apply(calculate_7d_persistence)
#     .reset_index(drop=True)
# )


# print("\n=== 7-DAY PERSISTENCE ===")

# print(source_dates_7d.head(20))

# print("\nPersistence distribution:")
# print(
#     source_dates_7d["persistence_7d"].describe()
# )
# --------------------------------------------------
# 11. INSPECT A HIGH-PERSISTENCE SOURCE
# --------------------------------------------------

# Find the source with the highest number of detection days
# top_source = (
#     source_table
#     .sort_values("detection_days", ascending=False)
#     .iloc[0]
# )

# lat = top_source["lat_grid"]
# lon = top_source["lon_grid"]

# print("Top source:")
# print("Latitude :", lat)
# print("Longitude:", lon)
# print("Detection days:", top_source["detection_days"])


# # Get all detection dates for this source
# top_dates = (
#     source_dates[
#         (source_dates["lat_grid"] == lat) &
#         (source_dates["lon_grid"] == lon)
#     ]
#     .sort_values("acq_date")
# )

# print("\nDetection dates:")
# print(top_dates.to_string(index=False))


# # Show our calculated 7-day persistence
# top_dates_7d = source_dates_7d[
#     (source_dates_7d["lat_grid"] == lat) &
#     (source_dates_7d["lon_grid"] == lon)
# ]

# print("\n7-day persistence:")
# print(top_dates_7d.to_string(index=False))
# --------------------------------------------------
# 12. FAST 7 / 30 / 90 DAY PERSISTENCE
# --------------------------------------------------

def max_persistence(dates, window_days):
    """
    Find the maximum number of distinct detection days
    occurring within any rolling time window.
    """

    dates = np.sort(dates.values).astype("datetime64[D]")

    if len(dates) == 1:
        return 1

    max_count = 1
    left = 0

    for right in range(len(dates)):

        while (
            dates[right] - dates[left]
            >= np.timedelta64(window_days, "D")
        ):
            left += 1

        count = right - left + 1

        if count > max_count:
            max_count = count

    return max_count


persistence_table = (
    source_dates
    .groupby(["lat_grid", "lon_grid"])["acq_date"]
    .agg(
        persistence_7d_max=lambda x: max_persistence(x, 7),
        persistence_30d_max=lambda x: max_persistence(x, 30),
        persistence_90d_max=lambda x: max_persistence(x, 90)
    )
    .reset_index()
)


print("\n=== PERSISTENCE TABLE ===")

print("Shape:", persistence_table.shape)

print(
    persistence_table[
        [
            "persistence_7d_max",
            "persistence_30d_max",
            "persistence_90d_max"
        ]
    ].describe()
)
# --------------------------------------------------
# 13. MERGE SOURCE + PERSISTENCE FEATURES
# --------------------------------------------------

candidate_source_features = source_table.merge(
    persistence_table,
    on=["lat_grid", "lon_grid"],
    how="left"
)

print("\n=== CANDIDATE SOURCE FEATURES ===")
print("Shape:", candidate_source_features.shape)

print("\nMissing values:")
print(candidate_source_features.isna().sum())
# --------------------------------------------------
# 14. SOURCE BEHAVIORAL FEATURES
# --------------------------------------------------

behavior_features = (
    firms
    .groupby(["lat_grid", "lon_grid"])
    .agg(
        night_ratio=("daynight", lambda x: (x == "N").mean()),
        type_0_ratio=("type", lambda x: (x == 0).mean()),
        type_2_ratio=("type", lambda x: (x == 2).mean()),
        type_3_ratio=("type", lambda x: (x == 3).mean())
    )
    .reset_index()
)


print("\n=== BEHAVIOR FEATURES ===")
print("Shape:", behavior_features.shape)
print(behavior_features.head())


# --------------------------------------------------
# 15. MERGE BEHAVIOR FEATURES
# --------------------------------------------------

candidate_source_features = candidate_source_features.merge(
    behavior_features,
    on=["lat_grid", "lon_grid"],
    how="left"
)


print("\n=== UPDATED FEATURE TABLE ===")
print("Shape:", candidate_source_features.shape)

print("\nMissing values:")
print(candidate_source_features.isna().sum())
# --------------------------------------------------
# 16. SAVE FIRMS FEATURE DATASET
# --------------------------------------------------

OUTPUT_PATH = (
    Path(__file__).parent
    / "firms_source_features_v1.csv"
)

candidate_source_features.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)
print("Final shape:", candidate_source_features.shape)