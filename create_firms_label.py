import pandas as pd

# ============================================================
# FIRMS HEURISTIC LABEL GENERATION — STRATEGY C
# ============================================================

INPUT_FILE = "firms_osm_features_v1.csv"
OUTPUT_FILE = "firms_labels_v1.csv"

print("=== LOADING FROZEN CHECKPOINT 2 DATASET ===")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)

# ------------------------------------------------------------
# LABEL RULES
# ------------------------------------------------------------
#
# y = 1:
# Persistent + elevated thermal intensity
# detection_days >= 3 AND max_frp >= 20
#
# y = 0:
# Short-lived + low thermal intensity
# detection_days == 1 AND max_frp < 10
#
# Everything else:
# Excluded from labeled training set
#
# IMPORTANT:
# These are FIRMS-only heuristic labels.
# OSM is NOT used to create the labels.
# ------------------------------------------------------------

positive_mask = (
    (df["detection_days"] >= 3)
    & (df["max_frp"] >= 20)
)

negative_mask = (
    (df["detection_days"] == 1)
    & (df["max_frp"] < 10)
)

# Sanity check: the two classes must never overlap
overlap = positive_mask & negative_mask

if overlap.any():
    raise RuntimeError(
        f"ERROR: Positive/negative overlap detected: {overlap.sum()} rows"
    )

# ------------------------------------------------------------
# Keep only confidently labeled rows
# ------------------------------------------------------------

labeled = df[positive_mask | negative_mask].copy()

labeled["y"] = -1

labeled.loc[positive_mask.loc[labeled.index], "y"] = 1
labeled.loc[negative_mask.loc[labeled.index], "y"] = 0

# Safety check
if (labeled["y"] == -1).any():
    raise RuntimeError("ERROR: Some rows were not assigned a binary label.")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

labeled.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print("\n=== LABELING COMPLETE ===")

print("Labeled shape:", labeled.shape)

print("\n=== CLASS COUNTS ===")
print(labeled["y"].value_counts().sort_index())

print("\n=== CLASS PERCENTAGES ===")
print(
    (labeled["y"].value_counts(normalize=True).sort_index() * 100)
    .round(3)
)

print("\n=== LABEL MEANING ===")
print("0 = Natural / vegetation-fire candidate")
print("1 = Industrial / persistent-thermal-source candidate")

print("\n=== EXCLUDED FROM LABELED SET ===")
excluded = len(df) - len(labeled)
print("Excluded rows:", excluded)
print("Excluded percentage:", round(excluded / len(df) * 100, 3), "%")

print("\nSaved:", OUTPUT_FILE)