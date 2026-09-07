import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Load clean OSM CSV
# --------------------------------------------------

osm_file = Path(r"Dataset\india_osm_industrial_quarry.csv")

osm = pd.read_csv(osm_file)

print("\n========== BASIC INFO ==========")
print("Shape:", osm.shape)

print("\n========== COLUMNS ==========")
print(osm.columns.tolist())

print("\n========== DATA TYPES ==========")
print(osm.dtypes)

print("\n========== FIRST 10 ROWS ==========")
print(osm.head(10).to_string())

print("\n========== MISSING VALUES ==========")
print(osm.isna().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicate rows:", osm.duplicated().sum())

print("\n========== DUPLICATE OSM IDs ==========")
print("Duplicate osm_id:", osm["osm_id"].duplicated().sum())

print("\n========== FACILITY TYPES ==========")
print(osm["facility_type"].value_counts(dropna=False))

print("\n========== FCODE ==========")
print(osm["fcode"].value_counts(dropna=False))

print("\n========== COORDINATE RANGE ==========")
print("Longitude:")
print("  min:", osm["longitude"].min())
print("  max:", osm["longitude"].max())

print("Latitude:")
print("  min:", osm["latitude"].min())
print("  max:", osm["latitude"].max())

print("\n========== UNIQUE STATES ==========")
print("Number of states:", osm["state"].nunique())
print(osm["state"].value_counts().head(20))

print("\n========== NAME COVERAGE ==========")
print("Named facilities:", osm["name"].notna().sum())
print("Unnamed facilities:", osm["name"].isna().sum())

print("\n========== SAMPLE FACILITY TYPES ==========")
print(
    osm.groupby("facility_type")
       .agg(
           records=("osm_id", "count"),
           unique_osm=("osm_id", "nunique"),
           named=("name", lambda x: x.notna().sum())
       )
)