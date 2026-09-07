import pandas as pd
import geopandas as gpd
import time
from pyproj import Geod
from shapely.ops import nearest_points
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

FIRMS_FILE = Path(r"Dataset\FIRMS_features\firms_source_points_v1.gpkg")
GPKG_FILE = Path(r"Dataset\india_osm_industrial_quarry.gpkg")

OUTPUT_FILE = "firms_osm_context_v1.csv"

PROJECTED_CRS = "EPSG:6933"

GEOD = Geod(ellps="WGS84")


# ==================================================
# FUNCTION: PROCESS ONE FACILITY TYPE
# ==================================================

def process_nearest(
    firms,
    facilities,
    facility_name
):

    print("\n")
    print("=" * 60)
    print(f"PROCESSING: {facility_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------
    # Project geometries
    # --------------------------------------------------

    print("Projecting geometries...")

    firms_projected = firms.to_crs(
        PROJECTED_CRS
    )

    facilities_projected = facilities.to_crs(
        PROJECTED_CRS
    )

    # --------------------------------------------------
    # Nearest spatial join
    # --------------------------------------------------

    print("Running nearest spatial join...")

    start = time.time()

    joined = gpd.sjoin_nearest(
        firms_projected,
        facilities_projected[
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
    join_time = time.time() - start

    print(
        f"Nearest join completed in {join_time:.2f} seconds"
    )

    print(
        "Rows returned by spatial join:",
        len(joined)
    )

    # --------------------------------------------------
    # HANDLE MULTIPLE EQUALLY-NEAREST POLYGONS
    # --------------------------------------------------

    print("\nChecking for multiple nearest matches...")

    duplicate_matches = (
        joined["source_id"].duplicated().sum()
    )

    print(
        "Extra duplicate matches:",
        duplicate_matches
    )

    if duplicate_matches > 0:

        print(
            "Keeping one nearest polygon per FIRMS source..."
        )

        joined = joined.sort_values(
            by=[
                "source_id",
                "projected_distance_m",
                "osm_id"
            ]
        )

        joined = joined.drop_duplicates(
            subset="source_id",
            keep="first"
        )

    print(
        "Rows after deduplication:",
        len(joined)
    )

    # --------------------------------------------------
    # Retrieve matched polygons
    # --------------------------------------------------

    print("Retrieving matched polygon geometries...")

    matched_polygons = facilities_projected.loc[
        joined["index_right"]
    ].geometry.reset_index(drop=True)

    matched_polygons.index = joined.index

    # --------------------------------------------------
    # Convert to WGS84
    # --------------------------------------------------

    print("Converting geometries to WGS84...")

    source_points_wgs84 = joined.geometry.to_crs(
        "EPSG:4326"
    )

    matched_polygons_wgs84 = gpd.GeoSeries(
        matched_polygons,
        crs=PROJECTED_CRS
    ).to_crs("EPSG:4326")

    # --------------------------------------------------
    # Geodesic distance
    # --------------------------------------------------

    print("Calculating geodesic distances...")

    start = time.time()

    distances = []

    for source_point, polygon in zip(
        source_points_wgs84,
        matched_polygons_wgs84
    ):

        if source_point is None or polygon is None:
            distances.append(None)
            continue

        _, closest_polygon_point = nearest_points(
            source_point,
            polygon
        )

        _, _, distance_m = GEOD.inv(
            source_point.x,
            source_point.y,
            closest_polygon_point.x,
            closest_polygon_point.y
        )

        distances.append(
            abs(distance_m)
        )

    distance_time = time.time() - start

    print(
        f"Distance calculation completed in "
        f"{distance_time:.2f} seconds"
    )

    # --------------------------------------------------
    # Build output
    # --------------------------------------------------

    prefix = facility_name.lower()

    result = pd.DataFrame({
        "source_id": joined["source_id"].values,

        f"nearest_{prefix}_distance_m":
            distances,

        f"nearest_{prefix}_osm_id":
            joined["osm_id"].values
    })

    # Add nearest facility type
    result[
        f"nearest_{prefix}_facility_type"
    ] = joined["facility_type"].values

    # Add state
    result[
        f"nearest_{prefix}_state"
    ] = joined["state"].values

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    print("\nDistance statistics:")

    print(
        result[
            f"nearest_{prefix}_distance_m"
        ].describe()
    )

    print(
        "\nZero-distance sources:",
        (
            result[
                f"nearest_{prefix}_distance_m"
            ] == 0
        ).sum()
    )

    # --------------------------------------------------
    # Return
    # --------------------------------------------------

    return result


# ==================================================
# 1. LOAD FIRMS
# ==================================================

print("\n========== LOADING FIRMS ==========")

firms = gpd.read_file(
    FIRMS_FILE,
    layer="firms_sources"
)

print(
    "FIRMS sources:",
    len(firms)
)


# ==================================================
# 2. LOAD INDUSTRIAL / QUARRY DATA
# ==================================================

print("\n========== LOADING INDUSTRIAL DATA ==========")

facilities = gpd.read_file(
    GPKG_FILE,
    layer="india_industrial_quarry"
)

print(
    "Total facilities:",
    len(facilities)
)


# ==================================================
# 3. SPLIT FACILITY TYPES
# ==================================================

industrial = facilities[
    facilities["facility_type"] == "Industrial"
].copy()

quarry = facilities[
    facilities["facility_type"] == "Quarry"
].copy()

print(
    "Industrial:",
    len(industrial)
)

print(
    "Quarry:",
    len(quarry)
)


# ==================================================
# 4. PROCESS INDUSTRIAL
# ==================================================

industrial_result = process_nearest(
    firms,
    industrial,
    "Industrial"
)


# ==================================================
# 5. PROCESS QUARRY
# ==================================================

quarry_result = process_nearest(
    firms,
    quarry,
    "Quarry"
)


# ==================================================
# 6. COMBINE RESULTS
# ==================================================

print("\n========== COMBINING RESULTS ==========")

# FIRMS attributes without geometry
firms_df = pd.DataFrame(
    firms.drop(columns="geometry")
)

final = firms_df.merge(
    industrial_result,
    on="source_id",
    how="left"
)

final = final.merge(
    quarry_result,
    on="source_id",
    how="left"
)


# ==================================================
# 7. VALIDATION
# ==================================================

print("\n========== FINAL VALIDATION ==========")

print(
    "Final shape:",
    final.shape
)

print(
    "Unique source IDs:",
    final["source_id"].nunique()
)

print(
    "Duplicate source IDs:",
    final["source_id"].duplicated().sum()
)

print("\nMissing values:")

print(
    final.isna().sum()
)


# ==================================================
# 8. SAVE
# ==================================================

print("\n========== SAVING ==========")

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print(
    "\n========== COMPLETE =========="
)