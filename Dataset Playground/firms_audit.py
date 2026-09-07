from pathlib import Path

import pandas as pd

FILE_PATH = Path(__file__).parent / "Dataset" / "fire_archive_SV-C2_793789.csv"

df = pd.read_csv(FILE_PATH)
print("Shape", df.shape)
print("\nColumns:")
print(df.columns.tolist())

#DataTypes
print("DataTypes")
print(df.dtypes)

print("Missing values")
print(df.isnull().sum())

print("Duplicated Rows")
print("Exact duplicates:", df.duplicated().sum())

print("Unique values")
for col in ["satellite", "instrument", "version", "confidence", "daynight", "type"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
    
print("Numeric Summary")
print(df[[
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "frp"
]].describe().T)

print("=== DATE RANGE ===")
print("First date:", df["acq_date"].min())
print("Last date :", df["acq_date"].max())

print("\n=== DATE SAMPLE ===")
print(df["acq_date"].head(10).tolist())

print("\n=== ACQUISITION TIME SAMPLE ===")
print(df["acq_time"].head(20).tolist())

print("\n=== UNIQUE ACQUISITION TIMES ===")
print("Unique times:", df["acq_time"].nunique())

print("\n=== RECORDS PER DAY ===")
daily_counts = df.groupby("acq_date").size()

print(daily_counts.describe())

print("\nDays with zero records:")
date_range = pd.date_range(
    start=df["acq_date"].min(),
    end=df["acq_date"].max(),
    freq="D"
)

missing_days = date_range.difference(
    pd.to_datetime(daily_counts.index)
)

print(missing_days.tolist())

print("\n=== TOP 10 BUSIEST DAYS ===")
print(daily_counts.sort_values(ascending=False).head(10))

print("\n=== 10 LOWEST-DETECTION DAYS ===")
print(daily_counts.sort_values().head(10))

print("=== GEOGRAPHIC RANGE ===")

print("Latitude:")
print("  Min:", df["latitude"].min())
print("  Max:", df["latitude"].max())

print("\nLongitude:")
print("  Min:", df["longitude"].min())
print("  Max:", df["longitude"].max())


print("\n=== INDIA ROUGH BOUNDING BOX ===")

india_mask = (
    (df["latitude"] >= 6) &
    (df["latitude"] <= 38) &
    (df["longitude"] >= 68) &
    (df["longitude"] <= 98)
)

print("Inside rough India box:", india_mask.sum())
print("Outside:", (~india_mask).sum())
print("Percentage inside:", round(india_mask.mean() * 100, 3))


print("\n=== LATITUDE DISTRIBUTION ===")
print(df["latitude"].describe())


print("\n=== LONGITUDE DISTRIBUTION ===")
print(df["longitude"].describe())


print("\n=== DETECTIONS BY LATITUDE BAND ===")

lat_bins = [6, 10, 15, 20, 25, 30, 35, 40]

lat_band = pd.cut(
    df["latitude"],
    bins=lat_bins
)

print(lat_band.value_counts().sort_index())


print("\n=== DETECTIONS BY LONGITUDE BAND ===")

lon_bins = [65, 70, 75, 80, 85, 90, 95, 100]

lon_band = pd.cut(
    df["longitude"],
    bins=lon_bins
)

print(lon_band.value_counts().sort_index())

print("=== FIRMS TYPE COUNTS ===")
print(df["type"].value_counts().sort_index())


print("\n=== TYPE PERCENTAGES ===")
print(
    df["type"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\n=== TYPE × DAY/NIGHT ===")
print(
    pd.crosstab(
        df["type"],
        df["daynight"],
        normalize="index"
    ).round(3)
)


print("\n=== TYPE × CONFIDENCE ===")
print(
    pd.crosstab(
        df["type"],
        df["confidence"],
        normalize="index"
    ).round(3)
)


print("\n=== FRP BY TYPE ===")
print(
    df.groupby("type")["frp"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


print("\n=== BRIGHTNESS BY TYPE ===")
print(
    df.groupby("type")["brightness"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


print("\n=== TYPE BY MONTH ===")

df["month"] = pd.to_datetime(df["acq_date"]).dt.month

print(
    pd.crosstab(
        df["month"],
        df["type"]
    )
)

print("=== THERMAL STATISTICS ===")

thermal_cols = [
    "brightness",
    "bright_t31",
    "frp"
]

print(df[thermal_cols].describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
).T)


print("\n=== THERMAL CONTRAST ===")

thermal_delta = df["brightness"] - df["bright_t31"]

print(thermal_delta.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
))


print("\n=== ZERO / EXTREME FRP ===")

print("FRP = 0:", (df["frp"] == 0).sum())
print("FRP > 50:", (df["frp"] > 50).sum())
print("FRP > 100:", (df["frp"] > 100).sum())
print("FRP > 200:", (df["frp"] > 200).sum())


print("\n=== EXTREME BRIGHTNESS ===")

print("Brightness < 250:", (df["brightness"] < 250).sum())
print("Brightness > 350:", (df["brightness"] > 350).sum())
print("Brightness > 360:", (df["brightness"] > 360).sum())


print("\n=== EXTREME BRIGHT_T31 ===")

print("bright_t31 < 250:", (df["bright_t31"] < 250).sum())
print("bright_t31 > 350:", (df["bright_t31"] > 350).sum())
print("bright_t31 > 400:", (df["bright_t31"] > 400).sum())

print("=== COORDINATE PRECISION ===")

print("Unique latitude values:", df["latitude"].nunique())
print("Unique longitude values:", df["longitude"].nunique())

print("\nSample coordinates:")
print(
    df[["latitude", "longitude"]]
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# APPROXIMATE 1 KM GRID
# --------------------------------------------------

df["lat_grid"] = df["latitude"].round(2)
df["lon_grid"] = df["longitude"].round(2)

grid_counts = (
    df.groupby(["lat_grid", "lon_grid"])
      .size()
      .sort_values(ascending=False)
)

print("\n=== 1 KM GRID ===")
print("Unique ~1 km grid cells:", len(grid_counts))

print("\nTop 20 grid cells:")
print(grid_counts.head(20))


# --------------------------------------------------
# HOW MANY DETECTIONS ARE IN REPEATED CELLS?
# --------------------------------------------------

cell_detection_counts = grid_counts

print("\n=== GRID PERSISTENCE ===")

print("Cells with >= 1 detection :", (cell_detection_counts >= 1).sum())
print("Cells with >= 2 detections :", (cell_detection_counts >= 2).sum())
print("Cells with >= 5 detections :", (cell_detection_counts >= 5).sum())
print("Cells with >= 10 detections:", (cell_detection_counts >= 10).sum())
print("Cells with >= 50 detections:", (cell_detection_counts >= 50).sum())
print("Cells with >= 100 detections:", (cell_detection_counts >= 100).sum())

print("\n=== GRID SIZE DISTRIBUTION ===")
print(
    cell_detection_counts.describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
)


print("=== SPATIAL + TEMPORAL PERSISTENCE ===")

# Make sure date is treated as a date
df["acq_date_dt"] = pd.to_datetime(df["acq_date"])


# Number of unique detection days per ~1 km cell
grid_persistence = (
    df.groupby(["lat_grid", "lon_grid"])["acq_date_dt"]
      .nunique()
      .sort_values(ascending=False)
)

print("\n=== UNIQUE DETECTION DAYS PER CELL ===")

print(grid_persistence.describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
))


print("\n=== PERSISTENT CELLS ===")

for threshold in [2, 5, 10, 30, 60, 90, 180]:
    count = (grid_persistence >= threshold).sum()
    print(f"Cells detected on >= {threshold:3} different days: {count}")


print("\n=== TOP 20 MOST PERSISTENT CELLS ===")

print(grid_persistence.head(20))

print("=== OBSERVATION GAP ANALYSIS ===")

# Convert dates
dates = pd.to_datetime(df["acq_date"])

# All calendar dates in the requested year
all_dates = pd.date_range(
    start="2024-01-01",
    end="2024-12-31",
    freq="D"
)

observed_dates = pd.DatetimeIndex(dates.unique())

missing_dates = all_dates.difference(observed_dates)

print("Total calendar days:", len(all_dates))
print("Days with detections:", len(observed_dates))
print("Days without detections:", len(missing_dates))

print("\nMissing dates:")
print(missing_dates.tolist())


# Detection count around each gap
daily_counts = dates.value_counts().sort_index()

print("\n=== NEIGHBOURING DAYS AROUND GAPS ===")

for gap in missing_dates:
    before = gap - pd.Timedelta(days=1)
    after = gap + pd.Timedelta(days=1)

    print(
        f"{gap.date()} | "
        f"before={daily_counts.get(before, 0)} | "
        f"after={daily_counts.get(after, 0)}"
    )