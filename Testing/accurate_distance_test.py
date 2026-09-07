import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from pyproj import Geod
import time
from pathlib import Path


# ==================================================
# FILES
# ==================================================

FIRMS_FILE = Path(r"Dataset\FIRMS_features\firms_source_points_v1.gpkg")
GPKG_FILE = Path(r"Dataset\india_osm_industrial_quarry.gpkg")

PROJECTED_CRS = "EPSG:6933"

# WGS84 ellipsoid for geodesic distance
GEOD = Geod(ellps="WGS84")


# ==================================================
# 1. LOAD FIRMS
# ==================================================

print("\n========== LOAD FIRMS ==========")

firms = gpd.read_file(
    FIRMS_FILE,
    layer="firms_sources"
)

firms_test = firms.head(1000).copy()

print("Test sources:", len(firms_test))


# ==================================================
# 2. LOAD INDUSTRIAL POLYGONS
# ==================================================

print("\n========== LOAD INDUSTRIAL DATA ==========")

industrial = gpd.read_file(
    GPKG_FILE,
    layer="india_industrial_quarry"
)

industrial = industrial[
    industrial["facility_type"] == "Industrial"
].copy()

print("Industrial polygons:", len(industrial))


# ==================================================
# 3. PROJECT
# ==================================================

print("\n========== PROJECTING ==========")

firms_projected = firms_test.to_crs(PROJECTED_CRS)

industrial_projected = industrial.to_crs(PROJECTED_CRS)

print("Projected CRS:", PROJECTED_CRS)


# ==================================================
# 4. FAST NEAREST JOIN
# ==================================================

print("\n========== FAST NEAREST JOIN ==========")

start = time.time()

nearest = gpd.sjoin_nearest(
    firms_projected,
    industrial_projected[
        [
            "osm_id",
            "facility_type",
            "name",
            "state",
            "geometry"
        ]
    ],
    how="left",
    distance_col="projected_distance_m"
)
print("\n========== JOINED COLUMNS ==========")
print(nearest.columns.tolist())

print("\n========== JOINED SAMPLE ==========")
print(nearest.head(3).to_string())

print(
    f"Join time: {time.time() - start:.2f} seconds"
)

print("Result shape:", nearest.shape)


# ==================================================
# 5. RECOVER MATCHED POLYGON GEOMETRY
# ==================================================

print("\n========== RECOVERING MATCHED POLYGONS ==========")

# sjoin_nearest stores the matching polygon's row
# number in index_right.
#
# Use that index to retrieve the actual polygon
# geometry from the projected industrial layer.

matched_polygons = industrial_projected.loc[
    nearest["index_right"]
].geometry.reset_index(drop=True)

# Make sure the order matches the FIRMS rows
matched_polygons.index = nearest.index

# ==================================================
# 6. CALCULATE EXACT GEODESIC DISTANCE
# ==================================================

print("\n========== CALCULATING GEODESIC DISTANCE ==========")

start = time.time()

# --------------------------------------------------
# Convert FIRMS points back to WGS84
# --------------------------------------------------

source_points_wgs84 = nearest.geometry.to_crs(
    "EPSG:4326"
)

# --------------------------------------------------
# Convert matched polygons back to WGS84
# --------------------------------------------------

matched_polygons_wgs84 = gpd.GeoSeries(
    matched_polygons,
    crs=PROJECTED_CRS
).to_crs("EPSG:4326")

# --------------------------------------------------
# Find closest point on each polygon
# --------------------------------------------------

accurate_distances = []

for source_point, polygon in zip(
    source_points_wgs84,
    matched_polygons_wgs84
):

    # Closest point on polygon boundary/interior
    _, closest_polygon_point = nearest_points(
        source_point,
        polygon
    )

    # Geodesic distance on WGS84 ellipsoid
    _, _, distance_m = GEOD.inv(
        source_point.x,
        source_point.y,
        closest_polygon_point.x,
        closest_polygon_point.y
    )

    accurate_distances.append(
        abs(distance_m)
    )

nearest["accurate_distance_m"] = accurate_distances

print(
    f"Distance calculation time: "
    f"{time.time() - start:.2f} seconds"
)

# ==================================================
# 7. COMPARE DISTANCES
# ==================================================

print("\n========== DISTANCE COMPARISON ==========")

comparison = nearest[
    [
        "projected_distance_m",
        "accurate_distance_m"
    ]
].copy()

comparison["difference_m"] = (
    comparison["accurate_distance_m"]
    - comparison["projected_distance_m"]
)

comparison["absolute_difference_m"] = (
    comparison["difference_m"].abs()
)

print(
    comparison.describe()
)

# ==================================================
# 8. RELATIVE DIFFERENCE
# ==================================================

comparison["relative_difference_percent"] = np.where(
    comparison["accurate_distance_m"] > 0,
    (
        comparison["absolute_difference_m"]
        / comparison["accurate_distance_m"]
    ) * 100,
    0
)

print("\n========== RELATIVE DIFFERENCE (%) ==========")

print(
    comparison[
        "relative_difference_percent"
    ].describe()
)

# ==================================================
# 9. LARGEST DIFFERENCES
# ==================================================

nearest["absolute_difference_m"] = (
    comparison["absolute_difference_m"].values
)

print("\n========== LARGEST DIFFERENCES ==========")

print(
    nearest[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "projected_distance_m",
            "accurate_distance_m",
            "absolute_difference_m"
        ]
    ]
    .sort_values(
        "absolute_difference_m",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# ==================================================
# 10. FINAL
# ==================================================

print("\n========== TEST COMPLETE ==========")