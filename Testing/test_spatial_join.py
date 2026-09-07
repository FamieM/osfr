import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

# ==================================================
# FILES
# ==================================================

FIRMS_FILE = Path(r"Dataset\FIRMS_features\firms_source_points_v1.gpkg")
GPKG_FILE = Path(r"Dataset\india_osm_industrial_quarry.gpkg")

# ==================================================
# 1. LOAD FIRMS
# ==================================================

print("\n========== LOAD FIRMS ==========")

firms = gpd.read_file(
    FIRMS_FILE,
    layer="firms_sources"
)

print("FIRMS sources:", len(firms))
print("CRS:", firms.crs)

# Take only 1,000 for the test
firms_test = firms.head(1000).copy()

print("Test sources:", len(firms_test))

# ==================================================
# 2. LOAD INDUSTRIAL GPKG
# ==================================================

print("\n========== LOAD INDUSTRIAL DATA ==========")

industrial = gpd.read_file(
    GPKG_FILE,
    layer="india_industrial_quarry"
)

print("Total facilities:", len(industrial))
print("CRS:", industrial.crs)

# ==================================================
# 3. SPLIT INDUSTRIAL / QUARRY
# ==================================================

industrial_only = industrial[
    industrial["facility_type"] == "Industrial"
].copy()

quarry_only = industrial[
    industrial["facility_type"] == "Quarry"
].copy()

print("\nIndustrial polygons:", len(industrial_only))
print("Quarry polygons:", len(quarry_only))

# ==================================================
# 4. PROJECT TO METRIC CRS
# ==================================================
#
# EPSG:6933 = World Equidistant Cylindrical / 
# equal-area style global projection.
#
# For this MVP test, it gives us metre-based
# geometry without using longitude/latitude degrees.
#
# We will validate the distance behavior before
# committing to the full dataset.
# ==================================================

PROJECTED_CRS = "EPSG:6933"

print("\n========== PROJECTING ==========")

firms_test = firms_test.to_crs(PROJECTED_CRS)

industrial_only = industrial_only.to_crs(PROJECTED_CRS)
quarry_only = quarry_only.to_crs(PROJECTED_CRS)

print("Projected CRS:", firms_test.crs)

# ==================================================
# 5. INDUSTRIAL NEAREST JOIN
# ==================================================

print("\n========== INDUSTRIAL NEAREST JOIN ==========")

start = time.time()

industrial_join = gpd.sjoin_nearest(
    firms_test,
    industrial_only[
        [
            "osm_id",
            "facility_type",
            "name",
            "state",
            "geometry"
        ]
    ],
    how="left",
    distance_col="nearest_industrial_distance_m"
)

elapsed = time.time() - start

print(f"Time: {elapsed:.2f} seconds")

print(
    "Result shape:",
    industrial_join.shape
)

# ==================================================
# 6. QUARRY NEAREST JOIN
# ==================================================

print("\n========== QUARRY NEAREST JOIN ==========")

start = time.time()

quarry_join = gpd.sjoin_nearest(
    firms_test,
    quarry_only[
        [
            "osm_id",
            "facility_type",
            "name",
            "state",
            "geometry"
        ]
    ],
    how="left",
    distance_col="nearest_quarry_distance_m"
)

elapsed = time.time() - start

print(f"Time: {elapsed:.2f} seconds")

print(
    "Result shape:",
    quarry_join.shape
)

# ==================================================
# 7. DISTANCE STATISTICS
# ==================================================

print("\n========== INDUSTRIAL DISTANCES ==========")

print(
    industrial_join[
        "nearest_industrial_distance_m"
    ].describe()
)

print("\n========== QUARRY DISTANCES ==========")

print(
    quarry_join[
        "nearest_quarry_distance_m"
    ].describe()
)

# ==================================================
# 8. SAMPLE RESULTS
# ==================================================

print("\n========== INDUSTRIAL SAMPLE ==========")

print(
    industrial_join[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "nearest_industrial_distance_m",
            "osm_id",
            "state"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n========== QUARRY SAMPLE ==========")

print(
    quarry_join[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "nearest_quarry_distance_m",
            "osm_id",
            "state"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ==================================================
# 9. ZERO-DISTANCE COUNTS
# ==================================================

print("\n========== ZERO DISTANCE ==========")

print(
    "Sources on/inside industrial polygon:",
    (
        industrial_join[
            "nearest_industrial_distance_m"
        ] == 0
    ).sum()
)

print(
    "Sources on/inside quarry polygon:",
    (
        quarry_join[
            "nearest_quarry_distance_m"
        ] == 0
    ).sum()
)

# ==================================================
# 10. EXTREME DISTANCES
# ==================================================

print("\n========== LARGEST INDUSTRIAL DISTANCES ==========")

print(
    industrial_join[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "nearest_industrial_distance_m"
        ]
    ]
    .sort_values(
        "nearest_industrial_distance_m",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\n========== LARGEST QUARRY DISTANCES ==========")

print(
    quarry_join[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "nearest_quarry_distance_m"
        ]
    ]
    .sort_values(
        "nearest_quarry_distance_m",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

print("\n========== TEST COMPLETE ==========")