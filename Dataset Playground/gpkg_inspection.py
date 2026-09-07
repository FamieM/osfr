import geopandas as gpd
from pathlib import Path

# --------------------------------------------------
# 1. File
# --------------------------------------------------

gpkg_file = Path(r"Dataset\india_osm_industrial_quarry.gpkg")

# --------------------------------------------------
# 2. List layers
# --------------------------------------------------

print("\n========== GPKG LAYERS ==========")

layers = gpd.list_layers(gpkg_file)

print(layers.to_string(index=False))

# --------------------------------------------------
# 3. Load the first layer
# --------------------------------------------------

layer_name = layers.iloc[0]["name"]

print("\nLoading layer:", layer_name)

gdf = gpd.read_file(
    gpkg_file,
    layer=layer_name
)

# --------------------------------------------------
# 4. Basic information
# --------------------------------------------------

print("\n========== BASIC INFO ==========")

print("Shape:", gdf.shape)

print("\n========== COLUMNS ==========")
print(gdf.columns.tolist())

print("\n========== DATA TYPES ==========")
print(gdf.dtypes)

print("\n========== CRS ==========")
print(gdf.crs)

print("\n========== GEOMETRY TYPE ==========")
print(gdf.geometry.geom_type.value_counts())

# --------------------------------------------------
# 5. Geometry validity
# --------------------------------------------------

print("\n========== GEOMETRY VALIDITY ==========")

print("Total geometries:", len(gdf))
print("Valid geometries:", gdf.geometry.is_valid.sum())
print("Invalid geometries:", (~gdf.geometry.is_valid).sum())
print("Missing geometries:", gdf.geometry.isna().sum())
print("Empty geometries:", gdf.geometry.is_empty.sum())

# --------------------------------------------------
# 6. Duplicate IDs
# --------------------------------------------------

print("\n========== OSM ID CHECK ==========")

print("Duplicate osm_id:", gdf["osm_id"].duplicated().sum())

# --------------------------------------------------
# 7. Facility types
# --------------------------------------------------

print("\n========== FACILITY TYPES ==========")

print(
    gdf["facility_type"]
    .value_counts(dropna=False)
)

# --------------------------------------------------
# 8. Sample records
# --------------------------------------------------

print("\n========== SAMPLE RECORDS ==========")

print(
    gdf[
        [
            "osm_id",
            "fcode",
            "facility_type",
            "name",
            "state"
        ]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# 9. Bounding box
# --------------------------------------------------

print("\n========== BOUNDING BOX ==========")

print(gdf.total_bounds)

# --------------------------------------------------
# 10. Geometry area statistics
# --------------------------------------------------
# NOTE:
# These are NOT meaningful physical areas yet if CRS
# is geographic (EPSG:4326). We only inspect them here.

print("\n========== GEOMETRY SIZE CHECK ==========")

print(gdf.geometry.area.describe())

# --------------------------------------------------
# 11. Coordinate columns if present
# --------------------------------------------------

if "longitude" in gdf.columns and "latitude" in gdf.columns:

    print("\n========== ATTRIBUTE COORDINATES ==========")

    print(
        gdf[["longitude", "latitude"]]
        .describe()
    )

# --------------------------------------------------
# 12. First geometry
# --------------------------------------------------

print("\n========== FIRST GEOMETRY ==========")

print(gdf.geometry.iloc[0])