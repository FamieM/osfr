import pandas as pd
import geopandas as gpd
from pathlib import Path

# ==================================================
# FILES
# ==================================================

FIRMS_FILE = Path(r"Dataset\FIRMS_features\firms_source_features_v1.csv")

# ==================================================
# 1. LOAD FIRMS FEATURES
# ==================================================

print("\n========== LOADING FIRMS ==========")

firms = pd.read_csv(FIRMS_FILE)

print("Shape:", firms.shape)

# ==================================================
# 2. CREATE REPRESENTATIVE SOURCE POINTS
# ==================================================
#
# lat_grid / lon_grid represent the spatial cell
# used to construct each FIRMS candidate source.
#
# We use the grid coordinate as the representative
# point for spatial contextual analysis.
# ==================================================

print("\n========== CREATING SOURCE POINTS ==========")

firms_gdf = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(
        firms["lon_grid"],
        firms["lat_grid"]
    ),
    crs="EPSG:4326"
)

# ==================================================
# 3. VERIFY
# ==================================================

print("\n========== GEODATAFRAME INFO ==========")

print("Shape:", firms_gdf.shape)

print("CRS:", firms_gdf.crs)

print(
    "Geometry types:"
)

print(
    firms_gdf.geometry.geom_type.value_counts()
)

print(
    "Missing geometry:",
    firms_gdf.geometry.isna().sum()
)

print(
    "Empty geometry:",
    firms_gdf.geometry.is_empty.sum()
)

# ==================================================
# 4. CHECK BOUNDS
# ==================================================

print("\n========== FIRMS SPATIAL BOUNDS ==========")

print(
    firms_gdf.total_bounds
)

# ==================================================
# 5. SAMPLE
# ==================================================

print("\n========== SAMPLE ==========")

print(
    firms_gdf[
        [
            "source_id",
            "lat_grid",
            "lon_grid",
            "detection_count",
            "detection_days",
            "persistence_7d_max",
            "geometry"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ==================================================
# 6. SAVE
# ==================================================
#
# We don't actually need to save this permanently
# yet, but saving a GeoPackage checkpoint makes
# debugging easier.
# ==================================================

OUTPUT_FILE = "firms_source_points_v1.gpkg"

firms_gdf.to_file(
    OUTPUT_FILE,
    layer="firms_sources",
    driver="GPKG"
)

print("\n========== SAVED ==========")

print(
    f"Saved to: {OUTPUT_FILE}"
)