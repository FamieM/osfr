import pandas as pd
import geopandas as gpd
from pathlib import Path

# ==================================================
# FILES
# ==================================================

CSV_FILE = Path(r"Dataset\india_osm_industrial_quarry.csv")
GPKG_FILE = Path(r"Dataset\india_osm_industrial_quarry.gpkg")

# ==================================================
# 1. LOAD DATA
# ==================================================

print("\n========== LOADING DATA ==========")

osm = pd.read_csv(CSV_FILE)

gdf = gpd.read_file(
    GPKG_FILE,
    layer="india_industrial_quarry"
)

print("CSV shape :", osm.shape)
print("GPKG shape:", gdf.shape)

# ==================================================
# 2. NORMALIZE OSM IDs
# ==================================================

print("\n========== NORMALIZING OSM IDs ==========")

osm["osm_id"] = osm["osm_id"].astype(str).str.strip()
gdf["osm_id"] = gdf["osm_id"].astype(str).str.strip()

print("CSV unique OSM IDs :", osm["osm_id"].nunique())
print("GPKG unique OSM IDs:", gdf["osm_id"].nunique())

# ==================================================
# 3. ID SET COMPARISON
# ==================================================

print("\n========== OSM ID MATCH ==========")

csv_ids = set(osm["osm_id"])
gpkg_ids = set(gdf["osm_id"])

csv_only = csv_ids - gpkg_ids
gpkg_only = gpkg_ids - csv_ids
common = csv_ids & gpkg_ids

print("Common IDs :", len(common))
print("CSV only  :", len(csv_only))
print("GPKG only :", len(gpkg_only))

# ==================================================
# 4. FACILITY TYPE CONSISTENCY
# ==================================================

print("\n========== FACILITY TYPE CONSISTENCY ==========")

csv_type = (
    osm[["osm_id", "facility_type"]]
    .drop_duplicates("osm_id")
    .set_index("osm_id")["facility_type"]
)

gpkg_type = (
    gdf[["osm_id", "facility_type"]]
    .drop_duplicates("osm_id")
    .set_index("osm_id")["facility_type"]
)

common_ids = csv_type.index.intersection(gpkg_type.index)

type_comparison = pd.DataFrame({
    "csv_type": csv_type.loc[common_ids],
    "gpkg_type": gpkg_type.loc[common_ids]
})

type_mismatch = (
    type_comparison["csv_type"] !=
    type_comparison["gpkg_type"]
)

print("Common records checked:", len(type_comparison))
print("Type mismatches:", type_mismatch.sum())

if type_mismatch.sum() > 0:
    print("\nExamples of mismatches:")
    print(
        type_comparison[type_mismatch]
        .head(10)
    )

# ==================================================
# 5. CREATE CSV POINTS
# ==================================================

print("\n========== CREATING CSV POINTS ==========")

csv_points = gpd.GeoDataFrame(
    osm.copy(),
    geometry=gpd.points_from_xy(
        osm["longitude"],
        osm["latitude"]
    ),
    crs="EPSG:4326"
)

# ==================================================
# 6. MATCH CSV POINT TO GPKG GEOMETRY
# ==================================================

print("\n========== CHECKING POINT/POLYGON RELATIONSHIP ==========")

# Only use matching IDs
csv_points = csv_points[
    csv_points["osm_id"].isin(common_ids)
].copy()

gpkg_match = gdf[
    gdf["osm_id"].isin(common_ids)
].copy()

# Match polygon geometry to each CSV point using osm_id
gpkg_match = (
    gpkg_match[
        ["osm_id", "geometry"]
    ]
    .rename(columns={"geometry": "polygon_geometry"})
)

check = csv_points.merge(
    gpkg_match,
    on="osm_id",
    how="inner"
)

# Create GeoDataFrame from the CSV points
check = gpd.GeoDataFrame(
    check,
    geometry="geometry",
    crs="EPSG:4326"
)

# ==================================================
# 7. POINT INSIDE POLYGON CHECK
# ==================================================

check["point_inside_polygon"] = check.geometry.within(
    check["polygon_geometry"]
)

print(
    "CSV coordinate point inside corresponding polygon:"
)
print(
    check["point_inside_polygon"]
    .value_counts()
)

# ==================================================
# 8. CORRECT POINT → POLYGON DISTANCE CHECK
# ==================================================

print("\n========== POINT → POLYGON DISTANCE ==========")

# Project CSV points to EPSG:3857
points_projected = check.to_crs("EPSG:3857")

# Convert polygon geometry column into a GeoSeries
# and project it separately
polygons_projected = gpd.GeoSeries(
    check["polygon_geometry"],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# Calculate distance correctly
points_projected["point_to_polygon_distance_m"] = (
    points_projected.geometry.distance(
        polygons_projected
    )
)

print(
    points_projected[
        "point_to_polygon_distance_m"
    ].describe()
)

# ==================================================
# 9. INSPECT THE 1,116 OUTLIERS
# ==================================================

print("\n========== POINTS OUTSIDE THEIR POLYGON ==========")

outside = points_projected[
    ~points_projected["point_inside_polygon"]
].copy()

print("Outside count:", len(outside))

print("\nDistance statistics for outside points:")

print(
    outside[
        "point_to_polygon_distance_m"
    ].describe()
)

print("\nLargest distances:")

print(
    outside[
        [
            "osm_id",
            "facility_type",
            "state",
            "longitude",
            "latitude",
            "point_to_polygon_distance_m"
        ]
    ]
    .sort_values(
        "point_to_polygon_distance_m",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)
# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n========== VALIDATION SUMMARY ==========")

print("CSV records:", len(osm))
print("GPKG records:", len(gdf))
print("Common OSM IDs:", len(common))
print("CSV-only IDs:", len(csv_only))
print("GPKG-only IDs:", len(gpkg_only))
print("Facility type mismatches:", type_mismatch.sum())

print(
    "Points inside corresponding polygons:",
    check["point_inside_polygon"].sum(),
    "/",
    len(check)
)

print("\nValidation complete.")